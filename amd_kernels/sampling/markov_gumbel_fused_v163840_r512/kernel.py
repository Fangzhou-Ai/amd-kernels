# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One Markov draft step of a Kimi-K3.1 DSpark draft in one launch, FlyDSL (gfx950).

256 workgroups of four waves; a workgroup owns 640 vocabulary columns (a wave ten 16-column tiles) for all rows
(<= 48, ``MT`` row tiles of 16). The step's embeddings ``w1[prev]`` sit in LDS (row pitch 1040 B) and, from there,
in registers as the MFMA's second operand; each wave streams its ``w2[v, :]`` rows once, one tile ahead, with the
weights as the first operand (16x16x32 bf16, so a lane owns four consecutive columns of one row). The wave's base
logits slab arrives in LDS by coalesced loads; per tile the fp32 sum is rounded to bf16 (the GEMV output), added to
the base logits with one bf16 rounding, written back to that slab, noised with vLLM's murmur3 Gumbel draw and folded
into the lane's maximum (first column on ties). After each half of its tiles the wave stores the slab into the draft
logits cache with 16-byte coalesced stores. The workgroup reduces its columns per row and publishes one
(value, column) pair per row with agent-scope stores; the last workgroup to take a ticket reduces the 256 pairs of
each row into ``draft_tokens`` and resets the ticket. The temperature picks the loop body of the whole wave
(uniform branch): no row noised, no division (every row T in {0, 1}), or the general one.
"""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.common import buffer_ops

from .reference import RANK, V

# _log_normal, _mix, _fmix, _noise, _xor_lane and _comb follow sampling/gumbel_partials_bias_bf16_v200064/kernel.py
F, I = fx.Float32, fx.Int32
NEG_INF = float("-inf")
SC1 = 0x10  # agent-scope (cross-XCD) stores / loads
REC_DW = 4  # dwords of a candidate record: value, column, tag, pad
RES_OFF = 16  # dwords
CNT_OFF = 1024
REC_OFF = 2048
NWG = 256
XCD_TILES = tuple(int(x) for x in __import__('os').environ.get('KK_XT', '11,9,11,9,11,9,11,9').split(','))  # tiles per wave by workgroup id % 8
TS = bool(int(__import__('os').environ.get('KK_TS', '0')))
TS_OFF = 2048 + 48 * 256 * 4
SPIN = int(__import__('os').environ.get('KK_SPIN', '1000000000'))
WG_COLS = V // NWG
ER = RANK // 2 + 4  # LDS row pitch in dwords
KSTEPS = RANK // 32
_C = [float(struct.unpack("<f", struct.pack("<f", 1.0 / k))[0]) for k in range(1, 9)]


def _fbits(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


_LN2_HI, _LN2_LO = _fbits(0x3F317217), _fbits(0x3377D1CF)


def _i32(x):
    return x - (1 << 32) if x >= (1 << 31) else x


def _log_normal(x):
    """Triton's ``tl.log`` (LLVM's llvm.log.f32 expansion on gfx950) for finite x >= 2^-126."""
    y = fx.math.log2(x, fastmath=fx.FastMathFlags.afn)
    r0 = y * F(_LN2_HI)
    f0 = fx.math.fma(y, F(_LN2_HI), r0 * F(-1.0))
    f1 = fx.math.fma(y, F(_LN2_LO), f0)
    return fx.math.fma(y, F(_LN2_HI), f1)


def _lshr(x, s):
    return (x >> s) & ((1 << (32 - s)) - 1)


def _rotl(x, s):
    return (x << s) | _lshr(x, 32 - s)


def _mix(h, key, key_c1=None):
    key = key * _i32(0xCC9E2D51) if key_c1 is None else key_c1
    key = _rotl(key, 15)
    key = key * _i32(0x1B873593)
    h = h ^ key
    h = _rotl(h, 13)
    return (h << 2) + h + _i32(0xE6546B64)


def _fmix(h):
    h = h ^ _lshr(h, 16)
    h = h * _i32(0x85EBCA6B)
    h = h ^ _lshr(h, 13)
    h = h * _i32(0xC2B2AE35)
    return h ^ _lshr(h, 16)


def _noise(h0, col_c1):
    """fp32 Gumbel noise of the vocab id whose product with 0xCC9E2D51 is ``col_c1``."""
    r = _fmix(_mix(h0, None, col_c1) ^ 16)
    u = _lshr(r, 16).to(F) * F(1.52587890625e-05) + ((r & 0xFFFF).to(F) + F(0.5)) * F(2.3283064365386963e-10)
    p = F(_C[7])
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = fx.math.fma(u, p, F(_C[k - 1]))
    series = (u * F(-1.0)) * p
    one_m = F(1.0) - u
    direct = _log_normal((one_m > F(5.960464477539063e-08)).select(one_m, F(5.960464477539063e-08)))
    l1 = (u < F(0.25)).select(series, direct)
    return _log_normal(l1 * F(-1.0)) * F(-1.0)


def _dpp(src, ctrl, old=None, bank_mask=0xF):
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    full = old is None and bank_mask == 0xF
    return I(_rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(src), ctrl, 0xF, bank_mask, full))


def _swap(op, v):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(v), fx.as_ir_value(v), False, False)
    return I(_llvm.extractvalue(T.i32, res, [0])), I(_llvm.extractvalue(T.i32, res, [1]))


def _xor_lane(v, m, upper):
    """Lane ``l ^ m`` of the wave as an i32; ``upper``: this lane's bit m is set (m = 16, 32)."""
    if m == 1:
        return _dpp(v, 0xB1)
    if m == 2:
        return _dpp(v, 0x4E)
    if m == 4:
        return _dpp(v, 0x104, old=_dpp(v, 0x114, bank_mask=0xA), bank_mask=0x5)
    if m == 8:
        return _dpp(v, 0x128)
    a, b = _swap(_rocdl_ops.permlane16_swap if m == 16 else _rocdl_ops.permlane32_swap, v)
    return upper.select(a, b)


def _comb(v1, i1, v2, i2):
    """Keep (v1, i1) iff v1 > v2 or (v1 == v2 and i1 < i2)."""
    keep = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return keep.select(v1, v2), keep.select(i1, i2)


def _atomic_add(addr, v):
    """Returning scalar global add (one active lane): the old value."""
    low = I(fx.rocdl.readfirstlane(T.i32, fx.Int64(addr).to(fx.Int32)))
    high = I(fx.rocdl.readfirstlane(T.i32, (fx.Int64(addr) >> 32).to(fx.Int32)))
    pointer = (low.to(fx.Uint32).to(fx.Uint64) | (high.to(fx.Uint32).to(fx.Uint64) << 32)).bitcast(fx.Int64)
    inc = I(fx.rocdl.readfirstlane(T.i32, I(v)))
    return I(_llvm.inline_asm(T.i32, [fx.as_ir_value(inc), fx.as_ir_value(pointer)],
                              "s_atomic_add $0, $2, 0 glc\n\ts_waitcnt lgkmcnt(0)", "=s,0,s,~{memory}",
                              has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT))


def _xcc_id():
    return I(_llvm.inline_asm(T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)\n\ts_waitcnt lgkmcnt(0)", "=s",
                              has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT)) & 15


def _bf16_lo(w):
    return (w << 16).bitcast(F)


def _bf16_hi(w):
    return (w & -65536).bitcast(F)




def _pick(vals, c):
    out = I(vals[-1])
    for i in range(len(vals) - 2, -1, -1):
        out = (c == i).select(I(vals[i]), out)
    return out


def _wave_or(b, upper16, upper32):
    for mm in (1, 2, 4, 8):
        b = b | _xor_lane(b, mm, upper16)
    b = b | _xor_lane(b, 16, upper16)
    b = b | _xor_lane(b, 32, upper32)
    return I(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(b)))


def compile_markov_chain(mt: int, ksteps: int, sleep: int = 1):
    MT = mt
    K = ksteps
    NW = 4
    TMAX = max(XCD_TILES)
    HMAXC = 2 * ((TMAX + 1) // 2)  # 8-column chunks per row of the larger half
    QMAX = 2 * TMAX
    pass
    PBD = TMAX * 8 + 4  # dwords per row of the wave's staging buffer (pitch 16 B aligned)
    ROWS = MT * 16
    assert 1 <= MT <= 3 and K >= 1 and NW * 32 * sum(XCD_TILES) * 16 == V
    name = f"markov_gumbel_chain_v163840_r512_mt{MT}_k{K}"

    @fx.struct
    class Shared:
        e: fx.Array[fx.Int32, ROWS * ER, 16]
        buf: fx.Array[fx.Int32, NW * ROWS * PBD, 16]
        st: fx.Array[fx.Int32, K * 3 * 64]  # per step: h0, temperature bits, cache row offset (-1: none), by row
        pv: fx.Array[fx.Int32, NW * ROWS]
        pi: fx.Array[fx.Int32, NW * ROWS]
        tk: fx.Array[fx.Int32, 64]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_base: fx.Int64,  # bf16 [n, k, V]: row stride i32_bs0 elements, step column s at s * V
        i32_bs0: fx.Int32,
        arg_w1: fx.Int64,  # bf16 [V, RANK]
        arg_w2: fx.Int64,  # bf16 [V, RANK]
        arg_ids: fx.Int64,  # int32 input_ids
        arg_anchor: fx.Int64,  # int64 [n]: step 0's previous token is input_ids[anchor[r]]
        arg_tokens: fx.Int64,  # int64 draft_tokens [*, i32_tok_stride]
        i32_tok_stride: fx.Int32,
        arg_idx: fx.Int64,  # int64 [n, i32_idx_stride]: request state index, < 0 invalid
        i32_idx_stride: fx.Int32,
        arg_pos: fx.Int64,  # int64 [n, i32_pos_stride]: sample_pos
        i32_pos_stride: fx.Int32,
        arg_temp: fx.Int64,  # fp32 [slots]
        arg_seed: fx.Int64,  # int64 [slots]
        arg_cache: fx.Int64,  # bf16 [rows, cache_cols, V]
        i32_cs0: fx.Int32,  # cache strides in elements
        i32_cs1: fx.Int32,
        arg_rowmap: fx.Int64,  # int32 [slots] cache row of a request, 0: the slot itself
        arg_sync: fx.Int64,  # generation, per-row result records, per-workgroup candidate records
        i32_n: fx.Int32,
        i32_pos_add: fx.Int32,  # low word added to sample_pos: 2^30 (vLLM's draft noise salt) - 1
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = I(fx.rocdl.readfirstlane(T.i32, tx // 64))
        smem = fx.SharedAllocator().allocate(Shared).peek()
        ids_r = buffer_ops.create_buffer_resource_from_addr(arg_ids, num_records_bytes=0x7FFFFFFF)
        anc_r = buffer_ops.create_buffer_resource_from_addr(arg_anchor, num_records_bytes=i32_n * 8)
        tok_r = buffer_ops.create_buffer_resource_from_addr(arg_tokens, num_records_bytes=0x7FFFFFFF)
        w1_r = buffer_ops.create_buffer_resource_from_addr(arg_w1, num_records_bytes=V * RANK * 2)
        idx_r = buffer_ops.create_buffer_resource_from_addr(arg_idx, num_records_bytes=0x7FFFFFFF)
        pos_r = buffer_ops.create_buffer_resource_from_addr(arg_pos, num_records_bytes=0x7FFFFFFF)
        tmp_r = buffer_ops.create_buffer_resource_from_addr(arg_temp, num_records_bytes=0x7FFFFFFF)
        sed_r = buffer_ops.create_buffer_resource_from_addr(arg_seed, num_records_bytes=0x7FFFFFFF)
        rmp_r = buffer_ops.create_buffer_resource_from_addr(arg_rowmap, num_records_bytes=0x7FFFFFFF)
        base_r = buffer_ops.create_buffer_resource_from_addr(arg_base, num_records_bytes=0x7FFFFFFF)
        cache_r = buffer_ops.create_buffer_resource_from_addr(arg_cache, num_records_bytes=0x7FFFFFFF)
        w2_r = buffer_ops.create_buffer_resource_from_addr(arg_w2, num_records_bytes=V * RANK * 2)
        sy_r = buffer_ops.create_buffer_resource_from_addr(arg_sync, num_records_bytes=0x7FFFFFFF)
        has_map = arg_rowmap != 0
        xcc = _xcc_id()
        rank = _atomic_add(arg_sync + fx.Int64(CNT_OFF * 4) + fx.Int64(xcc) * 128, 1) & (32 * NW - 1)  # this wave's index in its XCC
        tiles_w = I(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(_pick(XCD_TILES, xcc))))  # tiles of each wave
        pre_w = I(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(_pick(
            [32 * NW * sum(XCD_TILES[:c]) for c in range(8)], xcc))))
        T1 = (tiles_w + 1) // 2
        T2 = tiles_w - T1
        n0 = (pre_w + rank * tiles_w) * 16
        upper16 = (lane & 16) != 0
        upper32 = (lane & 32) != 0

        def load_tile(t, mask=None):
            wi = n0 + t * 16 + lane % 16
            return [Vec(buffer_ops.buffer_load(w2_r, wi * (RANK // 2) + k * 16 + (lane // 16) * 4, vec_width=4,
                                               dtype=I, mask=mask)) for k in range_constexpr(KSTEPS)]

        gen = I(buffer_ops.buffer_load(sy_r, 0, vec_width=1, dtype=I, cache_modifier=SC1))
        nxt = load_tile(0)

        # previous token of row ``lane`` (lane < n) for step 0
        okl = lane < i32_n
        a_idx = I(buffer_ops.buffer_load(anc_r, lane * 2, vec_width=1, dtype=I, mask=okl))
        tok_l = I(buffer_ops.buffer_load(ids_r, a_idx, vec_width=1, dtype=I, mask=okl))

        def build_state(s):
            """Row ``lane``'s Gumbel seed state, temperature and cache offset of step ``s`` into the LDS table."""
            slot = I(buffer_ops.buffer_load(idx_r, (lane * i32_idx_stride + s) * 2, vec_width=1, dtype=I, mask=okl))
            pos_lo = I(buffer_ops.buffer_load(pos_r, (lane * i32_pos_stride + s) * 2, vec_width=1, dtype=I,
                                              mask=okl)) + i32_pos_add
            valid = okl & (slot >= 0)
            slot0 = valid.select(slot, I(0))
            tmp = I(buffer_ops.buffer_load(tmp_r, slot0, vec_width=1, dtype=I, mask=valid)).bitcast(F)
            sd = Vec(buffer_ops.buffer_load(sed_r, slot0 * 2, vec_width=2, dtype=I, mask=valid))
            crow = has_map.select(I(buffer_ops.buffer_load(rmp_r, slot0, vec_width=1, dtype=I, mask=valid & has_map)),
                                  slot)
            cv = valid & (crow >= 0)
            smem.st.ptr[(s * 3 + 0) * 64 + lane] = _mix(_mix(_mix(I(0), I(sd[0])), I(sd[1]) ^ I(s)), pos_lo)
            smem.st.ptr[(s * 3 + 1) * 64 + lane] = valid.select(tmp, F(0.0)).bitcast(I)
            smem.st.ptr[(s * 3 + 2) * 64 + lane] = cv.select(crow * i32_cs0 + I(s) * i32_cs1, I(-1))

        def finish_step(smem, s, noisy, divide, afr, temps, h0s, cur):
            best_v = [F(NEG_INF) for _ in range(MT)]
            best_i = [I(0) for _ in range(MT)]
            last_step = s == K - 1
            for h in range_constexpr(2):
                init = [v.ir_value() for v in best_v] + [v.ir_value() for v in best_i] + [v.ir_value() for v in cur]
                for t, old in range(fx.Index(0), fx.Index(T1 if h == 0 else T2), fx.Index(1), init=init):
                    best_v = [F(x) for x in old[:MT]]
                    best_i = [I(x) for x in old[MT:2 * MT]]
                    wv = [Vec(x) for x in old[2 * MT:]]
                    ti = I(t) + (T1 if h == 1 else 0)
                    nt = ti + 1
                    more = nt < tiles_w
                    nt = more.select(nt, I(0))
                    nwi = n0 + nt * 16 + lane % 16
                    nwv = [Vec(buffer_ops.buffer_load(w2_r, nwi * (RANK // 2) + k * 16 + (lane // 16) * 4,
                                                      vec_width=4, dtype=I,
                                                      mask=None if not last_step else more)) for k in range_constexpr(KSTEPS)]
                    acc = [Vec.filled(4, 0.0, F) for _ in range(MT)]
                    for k in range_constexpr(KSTEPS):
                        for m in range_constexpr(MT):
                            acc[m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                                Vec.make_type(4, F), [wv[k].bitcast(fx.BFloat16),
                                                      afr[m * KSTEPS + k].bitcast(fx.BFloat16), acc[m], 0, 0, 0]))
                    cl = ti * 16 + (lane // 16) * 4  # this lane's first column inside the wave's slab
                    c0 = n0 + cl
                    for m in range_constexpr(MT):
                        off = wave * ROWS * PBD + (lane % 16 + 16 * m) * PBD + cl // 2
                        bw = [I(smem.buf.ptr[off]), I(smem.buf.ptr[off + 1])]
                        bias = [I(Vec.from_elements([F(acc[m][2 * e]), F(acc[m][2 * e + 1])], F).to(fx.BFloat16)
                                  .bitcast(I)[0]) for e in range_constexpr(2)]
                        ysum = []
                        for e in range_constexpr(2):
                            ysum.append(_bf16_lo(bw[e]) + _bf16_lo(bias[e]))
                            ysum.append(_bf16_hi(bw[e]) + _bf16_hi(bias[e]))
                        pk = [I(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(ysum[2 * i]), fx.as_ir_value(ysum[2 * i + 1])))
                              for i in range_constexpr(2)]
                        smem.buf.ptr[off] = pk[0]
                        smem.buf.ptr[off + 1] = pk[1]
                        tdiv = (temps[m] == F(0.0)).select(F(1.0), temps[m])
                        noisy_row = temps[m] != F(0.0)
                        kc = c0 * _i32(0xCC9E2D51)
                        for e in range_constexpr(4):
                            y = _bf16_lo(pk[e // 2]) if e % 2 == 0 else _bf16_hi(pk[e // 2])
                            v = (y / tdiv) if divide else y
                            if noisy:
                                v = v + noisy_row.select(_noise(h0s[m], kc + _i32((e * 0xCC9E2D51) & 0xFFFFFFFF)),
                                                         F(0.0))
                            gt = v > best_v[m]
                            best_v[m] = gt.select(v, best_v[m])
                            best_i[m] = gt.select(c0 + e, best_i[m])
                    if TS:
                        if (tx == 0) & (ti < 14):
                            t64 = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
                            buffer_ops.buffer_store(t64.to(fx.Int32), sy_r, TS_OFF + (bid * K + s) * 24 + 8 + ti)
                    result = yield ([v.ir_value() for v in best_v] + [v.ir_value() for v in best_i] +
                                    [v.ir_value() for v in nwv])
                best_v = [F(x) for x in result[:MT]]
                best_i = [I(x) for x in result[MT:2 * MT]]
                cur = [Vec(x) for x in result[2 * MT:]]
                # the half's cache slab, coalesced 16 B per lane
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                for it in range_constexpr((ROWS * HMAXC + 63) // 64):
                    j = lane + 64 * it
                    r, q = j // HMAXC, j % HMAXC
                    co = I(smem.st.ptr[(s * 3 + 2) * 64 + r % ROWS])
                    rm = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(atom, fx.slice(buf16, (None, (wave * ROWS * PBD + r * PBD + ((T1 * 8 if h == 1 else 0) + q * 4)) // 4)), rm)
                    buffer_ops.buffer_store(Vec(fx.memref_load_vec(rm)), cache_r,
                                            (co + n0 + (T1 * 16 if h == 1 else 0) + q * 8) // 2,
                                            mask=(j < ROWS * HMAXC) & (q < 2 * (T1 if h == 0 else T2)) & (co >= 0))
            for m in range_constexpr(MT):
                av, ai = best_v[m], best_i[m]
                for mm, up in ((16, upper16), (32, upper32)):
                    ov = _xor_lane(av.bitcast(I), mm, up).bitcast(F)
                    oi = _xor_lane(ai, mm, up)
                    av, ai = _comb(av, ai, ov, oi)
                if lane < 16:
                    smem.pv.ptr[(wave * MT + m) * 16 + lane] = av.bitcast(I)
                    smem.pi.ptr[(wave * MT + m) * 16 + lane] = ai
            return cur

        atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        lds16 = fx.logical_divide(fx.make_view(fx.recast_iter(fx.Int32, smem.e.ptr),
                                               fx.make_layout(ROWS * ER, 1)), fx.make_layout(4, 1))
        buf16 = fx.logical_divide(fx.make_view(fx.recast_iter(fx.Int32, smem.buf.ptr),
                                               fx.make_layout(NW * ROWS * PBD, 1)), fx.make_layout(4, 1))

        def a_frag(m, k):
            rm = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(atom, fx.slice(lds16, (None, ((lane % 16 + 16 * m) * ER + k * 16 + (lane // 16) * 4) // 4)), rm)
            return Vec(fx.memref_load_vec(rm))

        def stamp(s, stage):
            if TS:
                if tx == 0:
                    t64 = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
                    buffer_ops.buffer_store(t64.to(fx.Int32), sy_r, TS_OFF + (bid * K + s) * 24 + stage)

        if TS:
            if tx == 0:
                xcc = I(_llvm.inline_asm(T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s", has_side_effects=True,
                                         is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT))
                buffer_ops.buffer_store(xcc, sy_r, TS_OFF + (bid * K) * 24 + 6)
                buffer_ops.buffer_store(rank, sy_r, TS_OFF + (bid * K) * 24 + 22)
                buffer_ops.buffer_store(n0, sy_r, TS_OFF + (bid * K) * 24 + 23)
        stamp(0, 7)
        for s in range_constexpr(K):
            tag = gen + (s + 1)
            stamp(s, 0)
            # ---- prologue: w1[prev] rows and the base-logit slab into LDS
            ws = []
            for i in range_constexpr(ROWS // NW):
                row = wave + NW * i
                tok = I(fx.rocdl.readlane(T.i32, fx.as_ir_value(tok_l), fx.as_ir_value(row)))
                ws.append(Vec(buffer_ops.buffer_load(w1_r, tok * (RANK // 2) + lane * 4, vec_width=4, dtype=I,
                                                     mask=row < i32_n)))
            bch = []
            for it in range_constexpr((ROWS * QMAX + 63) // 64):
                j = lane + 64 * it
                r, q = j // QMAX, j % QMAX
                ok = (j < ROWS * QMAX) & (q < 2 * tiles_w) & (r < i32_n)
                bch.append((r, q, Vec(buffer_ops.buffer_load(base_r, (r * i32_bs0 + s * V + n0 + q * 8) // 2,
                                                             vec_width=4, dtype=I, mask=ok))))
            if s == 0:
                for sb in range_constexpr(K):
                    if wave == sb % NW:
                        build_state(sb)
            for it, (r, q, v) in enumerate(bch):
                if (lane + 64 * it) < ROWS * QMAX:
                    for j in range_constexpr(4):
                        smem.buf.ptr[wave * ROWS * PBD + r * PBD + q * 4 + j] = v[j]
            for i in range_constexpr(ROWS // NW):
                row = wave + NW * i
                for j in range_constexpr(4):
                    smem.e.ptr[row * ER + lane * 4 + j] = ws[i][j]
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            stamp(s, 1)
            afr = [a_frag(m, k) for m in range_constexpr(MT) for k in range_constexpr(KSTEPS)]
            temps = [I(smem.st.ptr[(s * 3 + 1) * 64 + lane % 16 + 16 * m]).bitcast(F) for m in range_constexpr(MT)]
            h0s = [I(smem.st.ptr[(s * 3 + 0) * 64 + lane % 16 + 16 * m]) for m in range_constexpr(MT)]
            flag = I(0)
            for m in range_constexpr(MT):
                gen_t = (temps[m] != F(0.0)) & (temps[m] != F(1.0))
                flag = flag | gen_t.select(I(3), (temps[m] != F(0.0)).select(I(1), I(0)))
            for mm in (1, 2, 4, 8):
                flag = flag | _xor_lane(flag, mm, upper16)
            flag = I(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(flag)))

            nxt_out = nxt
            if flag >= 2:
                nxt_out = finish_step(smem, s, True, True, afr, temps, h0s, nxt)
            else:
                if flag == 1:
                    nxt_out = finish_step(smem, s, True, False, afr, temps, h0s, nxt)
                else:
                    nxt_out = finish_step(smem, s, False, False, afr, temps, h0s, nxt)
            nxt = nxt_out
            fx.gpu.barrier()
            stamp(s, 2)

            # ---- publish this workgroup's candidate of every row; reduce a row; read the result
            if wave == 0:
                if tx < i32_n:
                    m_t, l_t = tx // 16, tx % 16
                    pv = [I(smem.pv.ptr[(w * MT + m_t) * 16 + l_t]).bitcast(F) for w in range_constexpr(NW)]
                    pi = [I(smem.pi.ptr[(w * MT + m_t) * 16 + l_t]) for w in range_constexpr(NW)]
                    v, i = pv[0], pi[0]
                    for w in range_constexpr(1, NW):
                        v, i = _comb(v, i, pv[w], pi[w])
                    buffer_ops.buffer_store(Vec.from_elements([v.bitcast(I), i, tag, I(0)], I), sy_r,
                                            REC_OFF + (tx * NWG + bid) * REC_DW, cache_modifier=SC1)
                stamp(s, 3)
                if bid < i32_n:
                    def rd():
                        return [Vec(buffer_ops.buffer_load(sy_r, REC_OFF + (bid * NWG + lane * 4 + j) * REC_DW,
                                                           vec_width=4, dtype=I, cache_modifier=SC1))
                                for j in range_constexpr(4)]

                    def bad_of(rs):
                        b = I(0)
                        for r_ in rs:
                            b = b | (I(r_[2]) != tag).select(I(1), I(0))
                        return b
                    rs = rd()
                    bad = _wave_or(bad_of(rs), upper16, upper32)
                    it = I(0)
                    while (bad != 0) & (it < SPIN):
                        if sleep:
                            fx.rocdl.s_sleep(sleep)
                        rs = rd()
                        bad = _wave_or(bad_of(rs), upper16, upper32)
                        it = it + 1
                    if (bad != 0) & (lane == 0):
                        buffer_ops.buffer_store(Vec.from_elements([I(s), bid, I(1), I(0)], I), sy_r, 8, cache_modifier=SC1)
                    bv, bi = F(NEG_INF), I(0)
                    for r_ in rs:
                        bv, bi = _comb(bv, bi, I(r_[0]).bitcast(F), I(r_[1]))
                    for mm, up in ((1, upper16), (2, upper16), (4, upper16), (8, upper16), (16, upper16), (32, upper32)):
                        ov = _xor_lane(bv.bitcast(I), mm, up).bitcast(F)
                        oi = _xor_lane(bi, mm, up)
                        bv, bi = _comb(bv, bi, ov, oi)
                    if lane == 0:
                        buffer_ops.buffer_store(Vec.from_elements([bv.bitcast(I), bi, tag, I(0)], I), sy_r,
                                                RES_OFF + bid * REC_DW, cache_modifier=SC1)
                stamp(s, 4)
                if (bid == 0) if s == K - 1 else (bid >= 0):
                    def rdres():
                        return Vec(buffer_ops.buffer_load(sy_r, RES_OFF + lane * REC_DW, vec_width=4, dtype=I,
                                                          cache_modifier=SC1, mask=okl))

                    def bad_res(rv):
                        return (okl & (I(rv[2]) != tag)).select(I(1), I(0))
                    rv = rdres()
                    bad2 = _wave_or(bad_res(rv), upper16, upper32)
                    it2 = I(0)
                    while (bad2 != 0) & (it2 < SPIN):
                        if sleep:
                            fx.rocdl.s_sleep(sleep)
                        rv = rdres()
                        bad2 = _wave_or(bad_res(rv), upper16, upper32)
                        it2 = it2 + 1
                    if (bad2 != 0) & (lane == 0):
                        buffer_ops.buffer_store(Vec.from_elements([I(s), bid, I(2), I(0)], I), sy_r, 12, cache_modifier=SC1)
                    smem.tk.ptr[lane] = I(rv[1])
                    if bid == 0:
                        if okl:
                            buffer_ops.buffer_store(Vec.from_elements([I(rv[1]), I(0)], I), tok_r,
                                                    (lane * i32_tok_stride + s) * 2)
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                stamp(s, 5)
            if s < K - 1:
                fx.gpu.barrier()
                tok_l = I(smem.tk.ptr[lane])
        if bid == 0:
            if tx == 0:
                buffer_ops.buffer_store(gen + K, sy_r, 0, cache_modifier=SC1)

    @flyc.jit
    def launch_markov_chain(
        arg_base: fx.Int64, i32_bs0: fx.Int32, arg_w1: fx.Int64, arg_w2: fx.Int64, arg_ids: fx.Int64,
        arg_anchor: fx.Int64, arg_tokens: fx.Int64, i32_tok_stride: fx.Int32, arg_idx: fx.Int64,
        i32_idx_stride: fx.Int32, arg_pos: fx.Int64, i32_pos_stride: fx.Int32, arg_temp: fx.Int64,
        arg_seed: fx.Int64, arg_cache: fx.Int64, i32_cs0: fx.Int32, i32_cs1: fx.Int32, arg_rowmap: fx.Int64,
        arg_sync: fx.Int64, i32_n: fx.Int32, i32_pos_add: fx.Int32, stream: fx.Stream,
    ):
        kernel(arg_base, i32_bs0, arg_w1, arg_w2, arg_ids, arg_anchor, arg_tokens, i32_tok_stride, arg_idx,
               i32_idx_stride, arg_pos, i32_pos_stride, arg_temp, arg_seed, arg_cache, i32_cs0, i32_cs1,
               arg_rowmap, arg_sync, i32_n, i32_pos_add).launch(grid=(NWG, 1, 1), block=(64 * NW, 1, 1),
                                                                stream=stream)

    return launch_markov_chain
