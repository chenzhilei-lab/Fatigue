# -*- coding: utf-8 -*-
"""提取式 2.3.2 (Kf 组合公式) 与 2.3.17 (全局应力梯度) 图片"""
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

# 找式 2.3.2 与 2.3.17 的段落（含 blip 的公式段）
targets = ['2.3.2', '2.3.3', '2.3.17', '2.3.18']
for i, p in enumerate(paras):
    texts = ''.join(t.text or '' for t in p.iter(W + 't')).strip()
    if any(t in texts for t in targets):
        blips = []
        for el in p.iter():
            if 'blip' in el.tag:
                rid = el.get(R + 'embed')
                if rid:
                    blips.append(rid2media.get(rid, '?'))
        if blips:
            print('[%d] %r blips=%s' % (i, texts[:60], blips))
            # 提取
            for b in blips:
                data = z.read('word/' + b)
                outname = '_fkm_eq_' + b.split('/')[-1]
                with open(outname, 'wb') as f:
                    f.write(data)
                print('   saved', outname, len(data), 'bytes')
