import json
def store_vars(vars):
    open('store/vars', 'w').write(str(vars))
    print('variables stored: ', len(vars))

def load_vars():
    print(open('store/vars').read())
    return json.load(open('store/vars').read())
