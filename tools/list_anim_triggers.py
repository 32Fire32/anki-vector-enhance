import anki_vector

try:
    with anki_vector.Robot() as r:
        triggers = getattr(r.anim, 'anim_trigger_list', None) or getattr(r.anim, 'anim_trigger_names', None) or []
        print('COUNT', len(triggers))
        for i, t in enumerate(triggers):
            print(f"{i+1}: {t}")
except Exception as e:
    print('ERROR', e)
