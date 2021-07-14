# Este arquivo contém as funções usadas para ajustar as curvas PV e outras funções úteis

############################################################### BIBLIOTECAS:
import numpy as np         # para fazer contas e mexer com matrizes
import pandas as pd        # para montar DataFrames (tabelas de bancos de dados)

from pathlib import Path   # para trabalhar com diretorios e arquivos
import pickle              # para gravar e ler dados

import matplotlib.pyplot as plt        # para gráficos
import seaborn as sns                  # para gráficos com DataFrames

from scipy.optimize import curve_fit   # para ajuste das curvas dos modelos

import math                            # para erf()
from scipy.interpolate import interp1d # para interpolar os pontos PV


############################################################### MODELOS:

# b = TLC (Total Lung Capacity)
def sigmoidvenegas1(x, a, b, c, d):
    return a + b/(1 + np.exp(-(x-c)/d))

########## paiva
def sigmoidpaiva(x,TLC,k1,k2):
    return TLC/(1+(k1*np.exp(-k2*x)))

# modificação nossa: incluindo offset no modelo Paiva
def sigmoidpaivaoffset1(x,TLC,k1,k2,offset):
    return TLC/(1+(k1*np.exp(-k2*x))) + offset

# baseado no artigo original do paiva1975, e incluindo offset:
def sigmoidpaivaoffset(x,TLC,k1,k2,offset):
    return TLC/(1+(k1*TLC*np.exp(-k2*x))) + offset

######### venegas2
def sigmoidvenegas2(x,TLC,B,k,c,d):
    return (TLC-(B*np.exp(-k*x)))/(1 + np.exp(-(x-c)/d))

# modificação nossa: incluindo offset
def sigmoidvenegas2offset(x,TLC,B,k,c,d,offset):
    return (TLC-(B*np.exp(-k*x)))/(1 + np.exp(-(x-c)/d)) + offset #alterei para funcionar como sigmoide.

# VM - Vm = TLC
######### murphy e engel
def sigmoidmurphy(x,VM,Vm,k1,k2,k3): ### CUIDADO: P = f(V) !!!
    return ( k1/(VM-x) ) + ( k2/(Vm-x) ) + k3

############################################################### FUNÇÕES:

'''
Carrega os arquivos .pickle das subpastas da pasta './porquinhos/'
e retorna um DataFrame com os dados.

As manobras C contém apenas 4 passos, e as D, apenas 5 passos.
'''
def carrega_pickles(folder = 'porquinhos'):
    dataframes_lst = [] # lista de dataframe: Cada elemento da lista corresponde a um dataframe de um porco/manobra/dados PV

    for file_name in Path(folder).rglob('*.pickle'):

        print(f"\rLendo {file_name.name}\t\t\t")

        with open(file_name, "rb") as file: # abre o arquivo.pickle

            porquinho = pickle.load(file)
            for manobra in porquinho: #Para cada manobra 

                if manobra == "D": # Posso fazer 3,4,5 passos
                    n_steps = 5
                elif manobra == "C": # Posso fazer 3,4 passos
                    n_steps = 4
                elif manobra == "B": # Posso fazer 3 passos
                    n_steps = 3

                # Formato os dados de entrada
                format_data = []

                for pi, pe, wi, we in zip(porquinho[manobra]["p_i"], porquinho[manobra]["p_e"],
                                          porquinho[manobra]["w_i"], porquinho[manobra]["w_e"]):

                    format_data.extend([pi,wi,pe,we])

                format_data = np.array(format_data).reshape(-1,2) # monta matriz de N linhas e 2 colunas


                ##########################################################
                caso = []
                caso.append(porquinho.name)
                caso.append(manobra)
                caso.append(format_data)
                caso.append(n_steps)
                casodf = pd.DataFrame(caso, index = ['Animal', 'Manobra', 'Dados', 'n_steps']).T
                dataframes_lst.append(casodf)    
    
    # Junta todos os dataframes da lista em um único DataFrame:
    dadosdf = pd.concat(dataframes_lst, ignore_index=True)
    
    # Extrai os dados de pressão e volume dos dados raw dos arquivos pickle:
    pv_lst = []
    for idx,caso in dadosdf.iterrows():
        pv = []
        ps,vs = Data2PV(caso.Dados)
        pv.append(ps)
        pv.append(vs)
        pvdf = pd.DataFrame([pv], columns = ['Pressoes', 'Volumes'])
        pv_lst.append(pvdf)
        
    pvdf_all = pd.concat(pv_lst, ignore_index=True)
    
    dadosdf_completo = pd.concat((dadosdf,pvdf_all),axis=1)
    
    # inclui uma coluna para volume esperado...
    dadosdf_completo["volume_esperado"] = 0
    
    return dadosdf_completo

'''
Retorna os vetores de pressão e volume a partir dos dados raw disponíveis nos pickles
'''
def Data2PV(data):
    data2 = data[0::2, :]
    pressures = data2[:,0]
    volumes = data2[:,1]
    return pressures,volumes

def encontra_volumes_limites_Murphy(parameters): ### no modelo de Murphy, P = f(V)
   
    # encontra limite superior:
    for volume_max in range(1,10000):
        pressure = sigmoidmurphy(volume_max,*parameters)
        if pressure > 100:
            break
            
    # encontra limite inferior:
    for volume_min in range(1,-10000,-1):
        pressure = sigmoidmurphy(volume_min,*parameters)
        if pressure < 0:
            break
            
    return volume_min,volume_max

# interpola vetores PV
# n_points = número de pontos intermediários
def interpola_PV(pressoes,volumes,n_points=0):
    if len(pressoes)<3:
        kind = "linear"
    elif len(pressoes)==3:
        kind = "quadratic"
    else:
        kind = "cubic"
    interp_pressures = np.linspace(pressoes[0], pressoes[-1], (len(pressoes)*(n_points+1))-n_points, endpoint=True)
    interp_func = interp1d(pressoes, volumes, kind=kind)
    interp_volumes = interp_func(interp_pressures)
    return interp_pressures, interp_volumes

# df: dataframe com os dados dos animais (pressão, volume, manobra etc)
# modelo: um dos modelos matemáticos definidos acima
# metodos: lm, dogbox, trf (parametros do curvefit)
# n_colunas: quantidade de colunas para disposição dos gráficos quando forem imprimidos
# texto: complemento do título
# TLC_index:Índice do array que refere-se ao parâmetro gerado equivalente TLC
# meus_bounds: condições de contorno (estimado)
# n_points_interp: número de pontos que serão interpolados
# debug: para printar resultados
# invert_PV: para inverter xdata e ydata no curvefit se aplicável (como no caso do modelo de Murhphy e Engel)
# limite_vol_max = 6000 e limite_vol_min = 100 - limites para melhor calcular o erro

def testa_modelo(df, modelo, meu_p0 = [], metodo = 'lm', n_colunas = 4, texto = '', TLC_index = 0, meus_bounds = [], n_points_interp=0, debug=True, invert_PV = False, df_final=None, limite_vol_max = 6000, limite_vol_min = 100):
    numero_de_casos = len(df)
    fig = plt.figure(figsize=(25,5*numero_de_casos/n_colunas))
    
    erro_vec = []
    n_fitted = 0
    
    for caso_teste in range(numero_de_casos):
        
        df.at[caso_teste, 'Modelo'] = modelo.__name__
        df.at[caso_teste, 'Metodo'] = metodo
        df.at[caso_teste, 'Ajustado'] = False
        df.at[caso_teste, 'Erro'] = 0
        df.at[caso_teste, "n_pontos_interpolacao"] = n_points_interp
        
        p_in = df.iloc[caso_teste].Pressoes
        v_in = df.iloc[caso_teste].Volumes
        
        # interpola pontos (se n_points_interp==0, a função não interpola)
        p, v = interpola_PV(p_in,v_in,n_points_interp)
        
        plt.subplot(int(numero_de_casos/n_colunas)+1,n_colunas,caso_teste+1)
        fig.tight_layout()
        if (n_points_interp > 0):
            plt.scatter(p,v,label='interp',c='k',marker='x')
        plt.scatter(p_in,v_in,label='raw')
        try:
            if (invert_PV == False): ################################### V = f(P)
                if (meu_p0 == []):                          # sem p0
                    if (meus_bounds == []): # sem bounds
                        parameters, pcov = curve_fit(modelo, p, v, method=metodo)
                    else:                   # com bounds
                        parameters, pcov = curve_fit(modelo, p, v, method=metodo, bounds=meus_bounds)
                else:                                       # com p0
                    if (meus_bounds == []): # sem bounds
                        parameters, pcov = curve_fit(modelo, p, v, method=metodo, p0 = meu_p0)
                    else:                   # com bounds
                        parameters, pcov = curve_fit(modelo, p, v, method=metodo, p0 = meu_p0, bounds=meus_bounds)
            else: ###################################################### P = f(V)
                if (meu_p0 == []):                          # sem p0
                    if (meus_bounds == []): # sem bounds
                        parameters, pcov = curve_fit(modelo, v, p, method=metodo)
                    else:                   # com bounds
                        parameters, pcov = curve_fit(modelo, v, p, method=metodo, bounds=meus_bounds)
                else:                                       # com p0
                    if (meus_bounds == []): # sem bounds
                        parameters, pcov = curve_fit(modelo, v, p, method=metodo, p0 = meu_p0)
                    else:                   # com bounds
                        parameters, pcov = curve_fit(modelo, v, p, method=metodo, p0 = meu_p0, bounds=meus_bounds)                    
            
            # Mostrar os parâmetros gerados pelo curvefit
            if debug:
                textop = ""
                for p in parameters:
                    if ( np.abs(p) > 1 ):
                        textop = textop + f'{p:7.1f}' + ' '
                    else:
                        textop = textop + f'{p:.3f}' + ' '
                print(f'Testando caso {caso_teste}: {df.iloc[caso_teste].Animal}: [{textop}]')
                
             # Para imprimir os gráficos   
            if (invert_PV == False): ################################### V = f(P)
                meu_p = range(1,100)
                meu_v = modelo(meu_p,*parameters)
            else: ###################################################### P = f(V) específico para o modelo de Murphy e Engel
                v_min,v_max = encontra_volumes_limites_Murphy(parameters)
                meu_v = np.asarray(range(v_min,v_max))
                meu_p = modelo(meu_v,*parameters)
            plt.plot(meu_p,meu_v,'r',label='fit')
            
            
            # Para o título do gráfico
            if ( df.iloc[caso_teste]["volume_esperado"] == 0 ):
                plt.title(f'Case: {df.iloc[caso_teste].Animal}. TLC = {parameters[TLC_index]:.0f} mL')
            
            else:
                v_esperado = df.iloc[caso_teste]["volume_esperado"]                
                if (modelo.__name__ == 'sigmoidmurphy'):
                    TLC = parameters[0] - parameters[1]
                else:
                    TLC = parameters[TLC_index]
                
                if (limite_vol_min <= TLC <= limite_vol_max ):
                    n_fitted = n_fitted + 1 #para contar os casos que fitaram
                    df.at[caso_teste, 'Ajustado'] = True
                
                    # Para cálculo do erro - TLC = TLC calculado pelo curvefit    
                    erro = 100*(TLC-v_esperado)/v_esperado
                    erro_vec.append(erro)
                    df.at[caso_teste, 'Erro'] = erro
                plt.title(f'Case: {df.iloc[caso_teste].Animal}. TLC = {TLC:.0f} mL. Error: {erro:.1f}% Steps:{df.iloc[caso_teste].n_steps}')
        except Exception as e:
            print(f'\tCaso {caso_teste} ({df.iloc[caso_teste].Animal}) deu erro... Steps:{df.iloc[caso_teste].n_steps}')
            plt.title(f'Case: {df.iloc[caso_teste].Animal}. Error fitting. Steps:{df.iloc[caso_teste].n_steps}')
      

        plt.xlabel('Pressure [cmH2O]')
        plt.ylabel('Volume [mL]')
        plt.legend()
    
    fig.suptitle(f'PV Graph. Model: {modelo.__name__}. {texto}', fontsize=16, y=1.05)
    plt.show()
    
    #Para cálculo do erro médio e norma do erro
    if ( len(erro_vec) > 0 ):
        erro_medio = np.mean(np.abs(erro_vec))
        erro_norm = np.linalg.norm(erro_vec)
    else:
        erro_medio = -1
        erro_norm = -1
        
    if df_final is not None:
        df_final = df_final.append({'modelo': modelo.__name__, 'metodo': metodo, 'norma_do_erro': erro_norm, 'erro_medio': erro_medio, 'ajustados': n_fitted, 'n_pontos_interpolacao': n_points_interp}, ignore_index=True)
        
    if debug:
        print(f'Norma(erro): {erro_norm:.1f}. Erro médio: {erro_medio:.2f}%. Ajustados: {n_fitted}.')
        
    
    return erro_norm, erro_medio, n_fitted, df_final
