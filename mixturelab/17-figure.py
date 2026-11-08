"""Deterministic SVG from all eight training-selected configurations."""
import html
from pathlib import Path
from .experiment import saved


def svg(report):
    rows=report['configurations'];w,h=600,1110;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="All eight mixture configurations, held-out and training density scores"><rect width="600" height="1110" fill="#fbfcf8"/><g fill="#193b36" font-family="sans-serif"><text x="30" y="38" font-size="13">MIXTURELAB / FIXED 106 / 36 / 36 SPLIT</text><text x="30" y="76" font-size="30">Better training is not better test.</text><text x="30" y="103" font-size="15">Mean negative log density, nats. Lower is better.</text>']
    colors=['#b3c4bc','#876d8e','#246859']
    for j,label in enumerate(['Train','Validation','Test']):
        parts.append(f'<rect x="{30+j*170}" y="125" width="12" height="12" fill="{colors[j]}"/><text x="{49+j*170}" y="137" font-size="14">{label}</text>')
    maxval=20.;start=165
    for i,r in enumerate(rows):
        y=start+i*105;chosen=r['id']==report['selected_id']
        text=f"{r['family']} / k={r['k']} / seed {r['seed']}"+(' / validation pick' if chosen else '')
        parts.append(f'<text x="30" y="{y}" font-size="16" font-weight="{600 if chosen else 400}">{html.escape(text)}</text>')
        for j,key in enumerate(['train_nll','validation_nll','test_nll']):
            value=r[key];yy=y+10+j*20;length=450*value/maxval
            parts.append(f'<rect data-id="{r["id"]}" data-metric="{key}" data-value="{value!r}" x="30" y="{yy}" width="{length!r}" height="12" fill="{colors[j]}"/><text x="{40+length!r}" y="{yy+11}" font-size="13">{value:.5f}</text>')
    parts.append('<path d="M30,1010H480" stroke="#93aaa0"/><text x="30" y="1033" font-size="13">0</text><text x="455" y="1033" font-size="13">20 nats</text><text x="30" y="1068" font-size="13">Same zero-based score scale for every bar. All eight retained.</text><text x="30" y="1090" font-size="13">UCI Wine, Aeberhard &amp; Forina (1992), CC BY 4.0.</text></g></svg>')
    return ''.join(parts)+'\n'

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];(root/'assets').mkdir(exist_ok=True)
    (root/'assets/density.svg').write_text(svg(saved()))
