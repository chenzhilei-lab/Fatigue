# -*- coding: utf-8 -*-
"""提取 FKM7 docx 公式 2.3.6-2.3.8 图片（用 R 命名空间 embed）"""
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
seen = set()
for i, p in enumerate(paras):
    texts = ''.join(t.text or '' for t in p.iter(W + 't'))
    if '2.3.6' in texts or '2.3.7' in texts or '2.3.8' in texts:
        for j in range(max(0, i - 1), min(i + 6, len(paras))):
            pj = paras[j]
            t2 = ''.join(t.text or '' for t in pj.iter(W + 't')).strip()
            for el in pj.iter():
                if 'blip' in el.tag:
                    rid = el.get(R + 'embed')
                    if rid and rid not in seen:
                        seen.add(rid)
                        media = rid2media.get(rid, '?')
                        data = z.read('word/' + media)
                        outname = '_fkm_eq_' + media.split('/')[-1]
                        with open(outname, 'wb') as f:
                            f.write(data)
                        print('para %d [%r] %s -> %s (%d bytes)' % (j, t2[:25], rid, media, len(data)))
print('done, extracted', len(seen), 'images')
