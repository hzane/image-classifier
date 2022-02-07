import json
from pathlib import Path
from fire import Fire
from tqdm import tqdm
from secrets import token_urlsafe

def main(
    ocrroot = 'ocrs/xarts6.train',
    labelroot = 'ocrs-annotations',
    prefix = 'xarts/xarts6.train/'
):
    ocrroot = Path(ocrroot)
    ocrs = list(ocrroot.rglob('*.json'))
    for item in tqdm(ocrs):
        ocr = json.loads(item.read_text())
        to = ocr['path'].replace(prefix, '') + ".json"
        results = []
        for idx, res in enumerate(ocr['res']):
            a, b, c, d = res['box']
            xmin = min(a[0], b[0], c[0], d[0])
            ymin = min(a[1], b[1], c[1], d[1])
            xmax = max(a[0], b[0], c[0], d[0])
            ymax = max(a[1], b[1], c[1], d[1])

            width = xmax-xmin
            height = ymax-ymin

            result = dict(
                x = xmin,
                y = ymin,
                width = width,
                height = height,
                attribute = '',
                valid = True,
                id = token_urlsafe(8),
                sourceID='',
                textAttribute='',
                order=idx+1,
            )
            results.append(result)
        step1 = dict(toolName='rectTool', result=results)
        obj = dict(width=0, height=0, valid=True, rotate=0, step_1 = step1)
        to = Path(labelroot, to)
        to.parent.mkdir(exist_ok=True, parents=True)
        to.write_text(json.dumps(obj, ensure_ascii = False, indent = 4))



if __name__ ==  '__main__':
    Fire(main)
