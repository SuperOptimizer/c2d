#!/usr/bin/env python3
"""Aggregate bench.tsv. Reports per-codec RD curves averaged across the full
corpus AND broken out by source category."""
import collections, sys, os, glob

# Build name -> category map by scanning testdata/<cat>/*
CATEGORIES = {}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for cat_dir in sorted(os.listdir(os.path.join(SCRIPT_DIR, 'testdata'))):
    full = os.path.join(SCRIPT_DIR, 'testdata', cat_dir)
    if not os.path.isdir(full): continue
    for f in os.listdir(full):
        if f.lower().endswith('.png') or f.lower().endswith('.jpg'):
            CATEGORIES[f] = cat_dir

def load(path):
    rows=[]
    with open(path) as f:
        hdr=f.readline().rstrip('\n').split('\t')
        for line in f:
            r=line.rstrip('\n').split('\t')
            if len(r) != len(hdr): continue
            d=dict(zip(hdr,r))
            try:
                d['bpp']=float(d['bpp'])
                d['psnr_db']=float(d['psnr_db'])
                d['bytes']=int(d['bytes'])
                if 'ssim' in d: d['ssim']=float(d['ssim'])
            except ValueError: continue
            d['source']=CATEGORIES.get(d['image'], '?')
            rows.append(d)
    return rows

def bci(rows, filter_source=None, metric='psnr_db'):
    o=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:
        if filter_source and r['source'] != filter_source: continue
        if metric not in r: continue
        o[r['codec']][r['image']].append((r['bpp'], r[metric]))
    for c in o:
        for i in o[c]: o[c][i].sort()
    return o

def interp(pts,bpp):
    if not pts or bpp<=pts[0][0] or bpp>=pts[-1][0]: return None
    for i in range(1,len(pts)):
        x0,y0=pts[i-1]; x1,y1=pts[i]
        if x0<=bpp<=x1:
            t=(bpp-x0)/(x1-x0)
            return y0+t*(y1-y0)
    return None

def report(by, bpps, label, codec_filter=None):
    codecs = sorted(by.keys())
    if codec_filter: codecs = [c for c in codecs if c in codec_filter]
    if not codecs: return
    # Restrict to images present in ALL codecs (apples-to-apples mean).
    common_imgs = None
    for c in codecs:
        s = set(by[c].keys())
        common_imgs = s if common_imgs is None else (common_imgs & s)
    n_img = len(common_imgs) if common_imgs else 0
    print(f"\n=== {label} ({n_img} images present in all codecs) ===")
    print(f"{'codec':<22} " + " ".join(f"{b:>7.2f}" for b in bpps) + "  n_present")
    for c in codecs:
        line = f"{c:<22} "
        for b in bpps:
            vals = []
            for img, pts in by[c].items():
                if common_imgs and img not in common_imgs: continue
                v = interp(pts, b)
                if v is not None: vals.append(v)
            if vals:
                line += f"{sum(vals)/len(vals):7.2f} "
            else:
                line += f"{'-':>7} "
        line += f"  {len(by[c])}"
        print(line)

def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else 'bench.tsv')
    if not rows:
        print("no data"); return
    sources = sorted(set(r['source'] for r in rows))
    print(f"Codecs: {sorted(set(r['codec'] for r in rows))}")
    print(f"Sources: {sources}")
    print(f"Total rows: {len(rows)}")
    bpps = [0.25, 0.5, 1.0, 2.0, 4.0]

    # Limit codec list for readability — focus on the most useful comparisons.
    focus_codecs = {'c2d-ycocg', 'c2di-ycocg', 'c2di-ycocg+bp', 'c2di-ycocg+bp+lo',
                    'jpeg', 'jpeg2k', 'jpegxl',
                    'webp', 'avif', 'heic', 'vvc'}

    by_all = bci(rows)
    report(by_all, bpps, "ALL IMAGES (mean PSNR @ bpp)", focus_codecs)
    for src in sources:
        by_src = bci(rows, filter_source=src)
        if not by_src: continue
        report(by_src, bpps, f"source={src} PSNR", focus_codecs)

    # If SSIM data present, also report SSIM tables.
    has_ssim = any('ssim' in r for r in rows)
    if has_ssim:
        print("\n\n############# SSIM tables (higher = better) #############")
        by_all_s = bci(rows, metric='ssim')
        report(by_all_s, bpps, "ALL IMAGES (mean SSIM @ bpp)", focus_codecs)
        for src in sources:
            by_src_s = bci(rows, filter_source=src, metric='ssim')
            if not by_src_s: continue
            report(by_src_s, bpps, f"source={src} SSIM", focus_codecs)

if __name__ == '__main__':
    main()
