import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
from mixturelab.experiment import saved
from mixturelab.figure import svg

class FigureTests(unittest.TestCase):
    def test_all_bars_are_exact_saved_scores_on_one_scale(self):
        report=saved();root=ET.fromstring(svg(report));bars=[n for n in root.iter() if 'data-metric' in n.attrib]
        self.assertEqual(len(bars),24)
        for node in bars:
            row=next(r for r in report['configurations'] if r['id']==node.attrib['data-id']);value=row[node.attrib['data-metric']]
            self.assertEqual(float(node.attrib['data-value']),value)
            self.assertEqual(float(node.attrib['width']),450*value/20)
            self.assertEqual(node.attrib['x'],'30')
    def test_source_svg_reproduction(self):
        # The installed wheel does not claim to bundle repository illustrations.
        path=Path(__file__).resolve().parents[1]/'assets/density.svg'
        self.assertEqual(path.read_text(),svg(saved()))

if __name__=='__main__':unittest.main()
