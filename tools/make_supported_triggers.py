import json
from pathlib import Path
root=Path(__file__).parent
triggers_file=root/'animation_triggers.json'
uns_file=root/'unsupported_triggers.json'
out_file=root/'supported_triggers.json'
tr=json.load(open(triggers_file,'r',encoding='utf-8'))
all_tr=[t['trigger'] if isinstance(t,dict) and 'trigger' in t else str(t) for t in tr.get('triggers',[])]
uns=json.load(open(uns_file,'r',encoding='utf-8'))['unsupported'] if uns_file.exists() else []
supported=[t for t in all_tr if t not in set(uns)]
open(out_file,'w',encoding='utf-8').write(json.dumps({'supported':supported,'total_doc':len(all_tr),'unsupported_count':len(uns)},indent=2,ensure_ascii=False))
print('Wrote',out_file,'Supported:',len(supported),'Total:',len(all_tr),'Unsupported:',len(uns))
