# -*- coding: utf-8 -*-
"""列出 FKM7 docx para 1318-1360 的文本与公式图片"""
import zipfile, re
import xml.etree.ElementTree as ET

z = zipfile.ZipFile(r'Standard\FKM_7th_ed_2020.docx')
xml = z.read('word/document.xml').decode('utf-8')
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
rels = z.read('word/_rels/document.xml.rels').decode('utf-8')
rid2media = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="(media/[^"]+)"', rels))
root = ET.fromstring(xml)
paras = [el for el in root.iter(W + 'p')]
for i in range(1318, 1365):
    p = paras[i]
    texts = ''.join(t.text or '' for t in p.iter(W + 't')).strip()
    blips = []
    for el in p.iter():
        if 'blip' in el.tag:
            rid = el.get(R + 'embed')
            if rid:
                blips.append(rid2media.get(rid, '?'))
    if texts or blips:
        print('[%d] %r blips=%s' % (i, texts[:85], blips))
