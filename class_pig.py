"""
É necessário importar essa classe para ler os arquivos porquinho*.pickle

Responsável por formatar o objeto Porco ... As funções com "double underscore" permitem trabalhar com objeto de forma um pouco mais amigavél.
"""

class pigClass:
    
    def __init__(self, name:str):
        self.name = name
        self._manobra = {}
        
    def __setitem__(self, key, value):
        self._manobra[key] = value
        
    def __getitem__(self,key):
        return self._manobra[key]
    
    def __iter__(self):
        return iter(self._manobra)
                
    def __repr__(self):
        
        data = ""
        for m in self._manobra.keys():
            data += f"{m}:\n"
            for d in self._manobra[m]:
                lst = self._manobra[m][d]
                data += f"\t{d} ({len(lst)}) = {lst}\n"
                

        return f"""Porco nome:{self.name}\n\n{''.join(data)}
                """