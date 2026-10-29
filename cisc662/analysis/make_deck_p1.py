"""
ANA-4: Presentation 1 deck (10 minutes, 8 slides + 1 backup) -> results/deck/P1.pptx. Numbers are read from results/db
where they exist (CE ablation ratio); the rest are the S1-S4 results quoted in docs/STATUS.md. Run with the deck env:
  /work/nvme/bffp/tmahmood1/cisc662/deckenv/bin/python -m cisc662.analysis.make_deck_p1
"""
import json
import os
import struct

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(PKG, "results", "figures")
DB = os.path.join(PKG, "results", "db")
OUT = os.path.join(PKG, "results", "deck")

NAVY, BLUE, ORANGE = RGBColor(0x14, 0x2B, 0x4D), RGBColor(0x2A, 0x7D, 0xE1), RGBColor(0xD9, 0x8C, 0x2B)
INK, MUTED, PALE, WHITE = RGBColor(0x22, 0x22, 0x22), RGBColor(0x66, 0x66, 0x66), RGBColor(0xEE, 0xF3, 0xFA), RGBColor(255, 255, 255)
FONT = "Calibri"
W, H = 13.333, 7.5


def ratio(group, num, den):
    a = json.load(open(os.path.join(DB, group, num + ".json")))["median_us"]
    b = json.load(open(os.path.join(DB, group, den + ".json")))["median_us"]
    return a / b


def txt(slide, x, y, w, h, paras, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    if isinstance(paras, str):
        paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(8)
        runs = p if isinstance(p, list) else [(p, {})]
        for t, o in runs:
            r = para.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.color.rgb = o.get("color", color)
    return tb


def title(slide, text):
    txt(slide, 0.6, 0.35, W - 1.2, 1.15, text, size=30, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE, name="Title")


def picture(slide, path, x, y, w, h):
    iw, ih = struct.unpack(">II", open(path, "rb").read(24)[16:24])
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    return slide.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def box(slide, x, y, w, h, fill=PALE, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def footer(slide, n):
    txt(slide, 0.6, H - 0.45, 9, 0.3, "CISC662 | Optimizing nanochat down to the assembly level", size=11, color=MUTED)
    txt(slide, W - 1.4, H - 0.45, 0.8, 0.3, str(n), size=11, color=MUTED, align=PP_ALIGN.RIGHT)


def new(prs, notes, ttl=None, n=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    if ttl:
        title(s, ttl)
    if n:
        footer(s, n)
    s.notes_slide.notes_text_frame.text = notes
    return s


def stat(slide, x, y, w, big, small, color=BLUE):
    box(slide, x, y, w, 1.35)
    txt(slide, x, y + 0.08, w, 0.7, big, size=34, color=color, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    txt(slide, x + 0.1, y + 0.78, w - 0.2, 0.55, small, size=13, color=MUTED, align=PP_ALIGN.CENTER)


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    approx = ratio("ce", "compile_ce_fwd_32768x32768", "ablation_approx_ce_fwd_32768x32768")

    # 1 title ---------------------------------------------------------------------------------------------------
    s = new(prs, "HAMZA (30 s). Hello. Our project takes a small program that trains a language model, finds the slowest "
            "pieces that run on the GPU, rewrites them by hand, and then reads the machine code the compiler produces, "
            "for NVIDIA and for AMD GPUs. This is the first of two presentations: today is what we are doing and the "
            "first results; the second one (December 17) is the final results.")
    bg = box(s, 0, 0, W, H, fill=NAVY, shape=MSO_SHAPE.RECTANGLE)
    txt(s, 0.9, 2.0, 11.5, 1.8, "Optimizing nanochat down to the assembly level", size=48, color=WHITE, bold=True)
    txt(s, 0.9, 3.9, 11.5, 0.8, "NVPTX and AMDGCN: rewriting the hot GPU kernels of an LLM trainer, then reading their machine code",
        size=22, color=RGBColor(0xCA, 0xDC, 0xFC))
    txt(s, 0.9, 5.6, 11.5, 0.9, ["Talha Mahmood and Hamza Mahmood", "CISC662 | Presentation 1 | November 2026"], size=18,
        color=WHITE)

    # 2 what and why --------------------------------------------------------------------------------------------
    s = new(prs, "HAMZA (1 min). nanochat is a small, complete program that trains a GPT-style language model on one GPU. "
            "Almost all of its time goes into a few kinds of GPU operations. Today PyTorch runs these either one at a time "
            "(eager) or fused automatically by torch.compile. Our question: can hand-written kernels beat both, and what "
            "does the machine code tell us about why? We go all the way down: PyTorch, Triton, CUDA, HIP, and finally the "
            "instructions: PTX and SASS for NVIDIA, and AMDGCN for AMD.",
            "What we are doing and why", 2)
    items = [("1", "Pick the hot spots", "Profile one training step of nanochat and find which GPU operations cost the most time."),
             ("2", "Rewrite them by hand", "Fused Triton kernels first, then CUDA and HIP versions of the three most important ones."),
             ("3", "Read the machine code", "Compile for NVIDIA (PTX, SASS) and AMD (AMDGCN); explain every speedup or slowdown by the instructions.")]
    for i, (n, h, b) in enumerate(items):
        y = 1.85 + i * 1.5
        c = box(s, 0.7, y, 0.9, 0.9, fill=BLUE, shape=MSO_SHAPE.OVAL)
        txt(s, 0.7, y, 0.9, 0.9, n, size=30, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        txt(s, 1.85, y - 0.05, 10.8, 0.45, h, size=24, color=NAVY, bold=True)
        txt(s, 1.85, y + 0.42, 10.8, 0.9, b, size=18)
    txt(s, 0.7, 6.35, 11.9, 0.5, "Baselines to beat: eager PyTorch and torch.compile (Inductor), measured on the same GPU in the same job.",
        size=16, color=MUTED)

    # 3 proposal vs reality -------------------------------------------------------------------------------------
    s = new(prs, "HAMZA (1 min). Honest status of the plan. The proposal named a V100 cluster as the main GPU. We cannot reach it "
            "from our allocation, so we use DeltaAI at NCSA: one GH200, an H100-class GPU. For AMD, DeltaAI has no AMD GPUs, so "
            "we already compile our kernels for three AMD chips and analyse the instructions without running them (static). "
            "Last week we confirmed access to NCSA Delta, which has MI100 and MI210 GPUs, so we can also measure on real AMD "
            "hardware. That node is in maintenance right now, so the measured AMD numbers are for the second presentation. "
            "Question you may get: why not DARWIN? Answer: we dropped it; everything is on NCSA systems now.",
            "Proposal versus reality: what changed and why", 3)
    rows = [(" ", "Proposal", "What we do now", "Status"),
            ("Main GPU", "DARWIN V100", "DeltaAI GH200 (H100-class)", "Measured"),
            ("AMD GPUs", "MI100 / MI250X", "AMDGCN compiled for gfx908, gfx90a, gfx942", "Static done"),
            ("AMD measured", "same", "NCSA Delta MI100 / MI210 (access confirmed)", "Node in maintenance"),
            ("CPU reference", "x86 EPYC", "Grace (aarch64) CPU, reported raw and normalised", "Script written"),
            ("V100 / A100", "primary / extra", "Static comparison of sm_70, sm_80, sm_90a SASS", "Planned")]
    colw = [2.1, 2.6, 5.4, 2.2]
    tbl = s.shapes.add_table(len(rows), 4, Inches(0.6), Inches(1.85), Inches(sum(colw)), Inches(4.3)).table
    for j, cw in enumerate(colw):
        tbl.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = tbl.cell(i, j)
            c.text = v
            p = c.text_frame.paragraphs[0]
            p.runs[0].font.size = Pt(16)
            p.runs[0].font.name = FONT
            p.runs[0].font.bold = (i == 0 or j == 0)
            p.runs[0].font.color.rgb = WHITE if i == 0 else INK
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (PALE if i % 2 else WHITE)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE

    # 4 method --------------------------------------------------------------------------------------------------
    s = new(prs, "TALHA or HAMZA (1 min). The method is a ladder. L0 is plain eager PyTorch. L1 is torch.compile. L2 is our Triton "
            "kernel. L3 is CUDA. L4 is HIP, the AMD version of CUDA. L5 is reading the instructions the compiler produced. "
            "Every kernel has to pass a correctness gate against a 64-bit reference before we are allowed to time it, and "
            "every timing uses GPU events, flushes the cache, and repeats in three separate processes with confidence "
            "intervals. That is how we make sure a speedup is real. Three kernels are on this ladder already: RMSNorm, "
            "QK-norm with RoPE, and softcap with cross-entropy.",
            "Method: a ladder from PyTorch down to instructions", 4)
    steps = [("L0", "Eager\nPyTorch"), ("L1", "torch.compile"), ("L2", "Triton\n(ours)"), ("L3", "CUDA\n(ours)"), ("L4", "HIP for AMD\n(ours)"),
             ("L5", "PTX / SASS /\nAMDGCN")]
    bw, gap, x0 = 1.8, 0.28, 0.6
    for i, (l, t) in enumerate(steps):
        x = x0 + i * (bw + gap)
        done = i <= 2
        box(s, x, 1.95, bw, 1.7, fill=BLUE if done else PALE)
        txt(s, x, 2.0, bw, 0.5, l, size=22, color=WHITE if done else NAVY, bold=True, align=PP_ALIGN.CENTER)
        txt(s, x, 2.55, bw, 1.0, t.split("\n"), size=16, color=WHITE if done else NAVY, align=PP_ALIGN.CENTER)
    txt(s, 0.6, 3.75, 12, 0.4, "Filled = done for our three kernels. We already read the NVIDIA and static AMD instructions for them (L5); CUDA and HIP versions (L3, L4) start next.", size=14, color=MUTED)
    rules = [("Correctness gate", "value and gradient checked against a 64-bit reference, and no worse than PyTorch's own error"),
             ("Timing rules", "GPU events, L2 cache flushed, 3 separate processes, bootstrap confidence intervals"),
             ("Fair comparison", "baselines run in the same job on the same GPU; only gated rows are plotted")]
    for i, (h, b) in enumerate(rules):
        x = 0.6 + i * 4.1
        box(s, x, 4.45, 3.9, 2.1)
        txt(s, x + 0.15, 4.55, 3.6, 0.5, h, size=20, color=NAVY, bold=True)
        txt(s, x + 0.15, 5.1, 3.6, 1.4, b, size=16)

    # 5 baseline ------------------------------------------------------------------------------------------------
    s = new(prs, "TALHA or HAMZA (1.3 min). This is the baseline we have to beat. We train a 12-layer model, one GPU, bf16. "
            "Stock nanochat with torch.compile and FlashAttention-3 takes 8.05 seconds per 100 steps; eager PyTorch takes "
            "15.08 seconds. So compile already gives about 1.9 times. The chart shows where one step's GPU time goes: matrix "
            "multiplications and attention dominate and are already near the hardware limit; those use vendor libraries "
            "and we do not try to beat them. What is left is the memory-bound operations: normalization, RoPE, the loss. "
            "The loss (softcap plus cross-entropy) is the largest one in eager at 29.6 milliseconds, and compile cuts it to "
            "4.5. Those are our targets.",
            "Baseline: compile already halves the step; the loss and norms are what remains", 5)
    picture(s, os.path.join(FIG, "s1_kernel_breakdown.png"), 0.5, 1.6, 8.4, 5.2)
    stat(s, 9.2, 1.75, 3.6, "8.05 s", "per 100 steps, torch.compile + FA3 (stock)")
    stat(s, 9.2, 3.3, 3.6, "15.08 s", "per 100 steps, eager PyTorch + FA3", color=MUTED)
    stat(s, 9.2, 4.85, 3.6, "37%", "model FLOP utilisation of the compiled baseline", color=ORANGE)

    # 6 results -------------------------------------------------------------------------------------------------
    s = new(prs, "TALHA or HAMZA (1.7 min). First results, all measured on the GH200, forward plus backward. Grey is speedup over "
            "eager PyTorch, orange is speedup over torch.compile, which is the harder baseline. Reading the chart from the "
            "bottom: the loss kernel, softcap plus cross-entropy, is 12 times faster than eager and 1.56 times faster than "
            "compile, and it uses 8 times less memory. QK-norm plus RoPE is 12 times over eager and 2 times over compile. "
            "The residual-mix plus norm kernel is 4.3 over eager and 1.1 over compile. Two honest points: for plain RMSNorm "
            "compile is already good, so we get only 1.18 times at 768 wide, and at 128 wide we tie compile (0.96, which is "
            "within noise). We report that because a result that is not better is still a result. All plotted kernels passed "
            "the correctness gates.",
            "First results: up to 2x over torch.compile and 12x over eager, with an honest tie", 6)
    picture(s, os.path.join(FIG, "p1_kernel_speedups.png"), 0.4, 1.55, 8.3, 5.3)
    pts = [[("Loss kernel: ", {"bold": True, "color": NAVY}), ("1.56x vs compile, 12.1x vs eager, 8x less memory (2.15 vs 17.2 GB)", {})],
           [("QK-norm + RoPE: ", {"bold": True, "color": NAVY}), ("2.08x vs compile", {})],
           [("RMSNorm: ", {"bold": True, "color": NAVY}), ("1.18x at 768 wide; a tie (0.96x) at 128 wide", {})],
           [("All kernels passed correctness gates.", {"color": MUTED})]]
    txt(s, 8.9, 1.8, 4.0, 4.9, pts, size=17)

    # 7 assembly finding ----------------------------------------------------------------------------------------
    s = new(prs, "TALHA (1.5 min). This is the kind of result the project is about: using the machine code to explain a number. "
            "Our loss kernel moves little data, so we expected it to be limited by memory. The instruction analysis "
            "says otherwise: each logit needs about 3.3 operations on the GPU's special-function unit, the "
            "hardware that computes exp and reciprocal, and those units are only a fraction of the arithmetic units. That "
            "gives a floor of about 0.95 milliseconds, higher than the 0.59 ms the memory floor allows. We measure 1.18 ms, "
            "close to that floor. So the next improvement is not about memory: it is about using fewer or cheaper exp "
            "instructions. As a check, a variant using the hardware's approximate tanh instruction is about "
            f"{approx:.1f} times faster than compile on the forward pass with a small error, but it is only an experiment; "
            "we have not adopted it. The profiler also shows our other kernels run at 82 to 98 percent of the GPU's "
            "measured memory bandwidth, so they are at the roof.",
            "The machine code explains it: the loss kernel is limited by exp units, not memory", 7)
    picture(s, os.path.join(FIG, "p1_ce_bounds.png"), 0.4, 1.6, 6.5, 4.0)
    picture(s, os.path.join(FIG, "p1_bandwidth.png"), 6.9, 1.6, 6.1, 4.0)
    txt(s, 0.6, 5.75, 12.2, 1.2, [[("Evidence: ", {"bold": True, "color": NAVY}),
        ("SASS has 33 EX2 + 20 RCP per 16 logits (about 3.3 special-function ops per logit). Compute floor 0.95 ms > memory floor 0.59 ms; "
         "measured 1.18 ms. Other kernels sit at 82-98% of the 3.65 TB/s measured bandwidth (Nsight Compute).", {})]], size=16)

    # 8 plan ----------------------------------------------------------------------------------------------------
    s = new(prs, "HAMZA (1.3 min). What comes next. Over the next month: the fused loss with the output layer so the huge logits "
            "tensor is never stored, a study of Triton matrix multiplication, an attention kernel, and CUDA and HIP versions "
            "of the three main kernels. In parallel the AMD side: static tables for all kernels, and measured runs on Delta "
            "as soon as the node is back. Then plug the kernels into nanochat training and check that the loss curve is the "
            "same, which is the gate that matters. The code freeze is December 8 and the second presentation is December 17. "
            "Budget: we have used about 0.85 service units (SU) of the 25 we planned, so cost is not the constraint. "
            "The main risk is AMD hardware availability; the fallback is the static analysis, which is already built.",
            "Plan to December 17, and the one real risk", 8)
    tl = [("Now to Nov 20", "Chunked loss + output layer, Triton GEMM and attention study, CUDA RMSNorm / RoPE"),
          ("Nov 20 to Dec 8", "HIP ports, AMD tables, measured MI100/MI210 runs, plug into training and check the loss curve"),
          ("Dec 8", "Code freeze, then final measurements"),
          ("Dec 17", "Presentation 2: results")]
    for i, (d, t) in enumerate(tl):
        y = 1.85 + i * 1.0
        box(s, 0.7, y, 2.6, 0.8, fill=BLUE)
        txt(s, 0.7, y, 2.6, 0.8, d, size=18, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        txt(s, 3.5, y, 5.3, 0.9, t, size=16, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 9.2, 1.85, 3.6, 2.0)
    txt(s, 9.3, 1.9, 3.4, 0.5, "Budget", size=20, color=NAVY, bold=True)
    txt(s, 9.3, 2.45, 3.4, 1.3, ["About 0.85 of 25 planned", "service units (SU) used"], size=16)
    box(s, 9.2, 4.05, 3.6, 2.4)
    txt(s, 9.3, 4.1, 3.4, 0.5, "Main risk", size=20, color=NAVY, bold=True)
    txt(s, 9.3, 4.65, 3.4, 1.8, "AMD node in maintenance. Fallback: static AMDGCN analysis, already built.", size=16)

    # 9 backup --------------------------------------------------------------------------------------------------
    s = new(prs, "BACKUP. Use only for questions. MUFU = multi-function unit (exp, reciprocal, tanh hardware). ISA = instruction set "
            "architecture. SASS = the real NVIDIA machine code; PTX is the portable layer above it. AMDGCN = AMD's machine code. "
            "Roofline = a chart of how close a kernel is to the hardware's memory or compute limit. Wavefront = AMD's 64-thread "
            "group, a warp is NVIDIA's 32. G1/G2 are our correctness gates: G1 for values, G2 for gradients.",
            "Backup: terms and numbers", 9)
    gl = [("MUFU", "special-function unit: exp, reciprocal, tanh hardware"), ("ISA / SASS / PTX", "machine code; SASS is NVIDIA's real code, PTX the portable layer"),
          ("AMDGCN", "AMD's machine code; 64-thread wavefronts, MFMA matrix instructions"),
          ("Roofline", "chart of achieved speed against the memory and compute limits"),
          ("Gates G1 / G2", "value and gradient correctness against a 64-bit reference"),
          ("R12", "the test setup: 12 layers, sequence 2048, 32k tokens per step, bf16")]
    for i, (a, b) in enumerate(gl):
        y = 1.8 + i * 0.82
        txt(s, 0.7, y, 3.0, 0.7, a, size=18, color=NAVY, bold=True)
        txt(s, 3.8, y, 9.0, 0.7, b, size=17)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "P1.pptx")
    prs.save(path)
    return path


if __name__ == "__main__":
    print(build())
