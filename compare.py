"""Compare saved vanilla, RL-only, and RLCD evaluation summaries."""
import argparse
import json
from pathlib import Path

ROOT = Path("/pfs/hyx/videojev-rlcd")
COLORS = {"vanilla": "#666666", "rl_only": "#d08000", "rlcd": "#0068b5"}


def draw(series, path, metric):
    width, height = 670, 460
    x0, x1, y0, y1 = 70, 630, 40, 390
    px = lambda x: x0 + x * (x1-x0)
    py = lambda y: y1 - y * (y1-y0)
    items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18">{metric}</text>',
             f'<path d="M{x0},{y0} V{y1} H{x1}" fill="none" stroke="#333" stroke-width="2"/>']
    if metric == "Reliability":
        items.append(f'<path d="M{px(0)},{py(0)} L{px(1)},{py(1)}" stroke="#aaa" stroke-dasharray="5 5"/>')
    for name, points in series.items():
        if not points:
            continue
        color = COLORS[name]
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x,y in points)
        items.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        items.extend(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{color}"/>' for x,y in points)
    for i,(name,color) in enumerate(COLORS.items()):
        items += [f'<line x1="{x0+i*180}" y1="425" x2="{x0+24+i*180}" y2="425" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{x0+30+i*180}" y="429" font-size="12">{name}</text>']
    items.append("</svg>")
    path.write_text("\n".join(items), encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    for name in COLORS:
        p.add_argument("--"+name.replace("_","-"), required=True)
    p.add_argument("--output", default=str(ROOT/"comparisons"))
    args=p.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    summaries={}
    reliability={}
    selective={}
    for name in COLORS:
        summary=json.loads(Path(getattr(args,name)).read_text())
        summaries[name]={k:v for k,v in summary.items() if k not in ("explicit","native")}
        reliability[name]=[(r["mean_confidence"],r["accuracy"]) for r in summary["explicit"]["table"] if r["count"]]
        selective[name]=[(r["coverage_all"],r["selective_accuracy"]) for r in summary["explicit"]["risk_coverage"]
                         if r["selective_accuracy"] is not None]
        summaries[name].update({k:summary["explicit"][k] for k in ("brier","ece","nll")})
    draw(reliability,out/"reliability_comparison.svg","Reliability")
    draw(selective,out/"risk_coverage_comparison.svg","Risk coverage")
    (out/"comparison.json").write_text(json.dumps(summaries,indent=2)+"\n")


if __name__=="__main__":
    main()
