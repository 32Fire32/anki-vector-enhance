import json
from pathlib import Path
ROOT=Path(__file__).parent
SUP=ROOT/'supported_triggers.json'
MAP=ROOT/'emotion_animation_mapping.json'
BACK=ROOT/'emotion_animation_mapping.json.bak'
if SUP.exists():
    supported=set(json.load(open(SUP,'r',encoding='utf-8'))['supported'])
else:
    supported=set()

mapping=json.load(open(MAP,'r',encoding='utf-8'))
# Backup
open(BACK,'w',encoding='utf-8').write(json.dumps(mapping,indent=2,ensure_ascii=False))
print('Backup written to',BACK)

fallbacks={
 'joy':['AlreadyAtFace','ComeHereSuccess','BlackJack_VictorWin','ConnectWakeUp'],
 'sadness':['ConnectToCubeFailure','ChargerDockingRequestGetout','BlackJack_VictorLose'],
 'confusion':['AudioOnlyHuh','ConnectToCubeGetIn','ConnectToCubeLoop'],
 'curiosity':['ExploringQuickScan','ExploringLookAround','ConnectToCubeGetIn','ExploringReactToHandLift'],
 'thinking':['CountingFastLoop','CountingGetInEven','ConnectToCubeGetIn','ConnectToCubeLoop'],
 'greeting':['ConnectWakeUp','ComeHereStart','ComeHereSuccess'],
 'thanks':['CubePounceWinSession','BlackJack_VictorWin'],
 'celebration':['BlackJack_VictorBlackJackWin','CubePounceWinSession'],
 'satisfied':['CubePounceWinSession','AlreadyAtFace'],
 'anger':['DriveLoopAngry','DriveStartAngry'],
 'surprise':['ConnectWakeUp','ExploringHuhClose','ExploringHuhFar']
}

newmap={}
for emo,conf in mapping.items():
    cand=conf.get('candidates',{})
    # Keep supported ones
    new_cand={k:v for k,v in cand.items() if k in supported}
    # If none left, fill from fallbacks keeping weights
    if not new_cand:
        fb=fallbacks.get(emo, list(supported)[:3])
        new_cand={k:2.0 for k in fb if k in supported}
    newmap[emo]={'candidates':new_cand,'probability':conf.get('probability',0.5),'intensity_variants':{}}

open(MAP,'w',encoding='utf-8').write(json.dumps(newmap,indent=2,ensure_ascii=False))
print('Wrote updated mapping to',MAP)
print('Sample:', json.dumps({k:list(v['candidates'].keys()) for k,v in newmap.items()},indent=2))
