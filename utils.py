from typing import Any

def kind_file(x:str)->bool:

    if str(x).rsplit('.',1)[1]=='txt':
        return True
    else:
        return False