import pandas as pd
import numpy as np



def prep_data(data):
     '''
     Intake a pandas dataframe or dictionary of two series or lists of type categorical which 
     describe the occurence of categories(1st) inside the entities (2nd) of a bi-parite graph. 
     A 3rd numerical column which describes the degree of the relationship of the category with 
     the entity is optional. 
    
     Parameters
     ----------
     data: pandas dataframe or dicionary of two series or lists of type categorical

     Returns
     -------
     adj_df: adjacency matrix as an n(# of categories) by m(# of entities) numpy array. 
     '''

     if isinstance(data, pd.DataFrame):
          pass
     elif isinstance(data, dict):
         if all(isinstance(col, list) for col in data.values()):
            data = pd.DataFrame(data)
         else: 
            value_types = [type(x) for x in data.values()]
            ValueError(f'Dictionary values must be lists. Got types {value_types} instead')
     else:
         raise TypeError(f'data must be a pandas dataframe or dictionary. Got type {type(data)} instead')


     if len(data.columns) < 3:
         data['value'] = 1
         col_names = data.columns 
         adj_df = data.pivot(index = col_names[0], columns = col_names[1], values = col_names[2]).fillna(0)

     return adj_df
     
 
def filter_df(data):
    cat_sums = data.groupby(data[0])[data[2]].sum()
    entity_sums = data.groupby(data[1])[data[2]].sum()
    total_sums = data[2].sum()

    data['rca_num'] = data[2] / data[1].map(entity_sums)
    data['rca_denom'] = data[0].map(cat_sums) / total_sums
    data['rca'] = data['rca_num'] / data['rca_denom'] 

    filtered_data = data[data['rca'] >= 1].drop(columns = ['rca_num', 'rca_denom', 'rca'])

    return filtered_data

            

def co_occurence(data):
    co_occ_dict = {}
    unique = sorted(set(data[0]))
    for cat_i in unique:
        entities_i = set(data[1][data[0] == cat_i])
        column = []
        for cat_j in unique:
            if cat_i == cat_j:
                column.append(1)
            else:
                entities_j = set(data[1][data[0] == cat_j])
                shared = entities_i & entities_j
                cond_prob = len(shared) / max(len(entities_i), len(entities_j))
                column.append(cond_prob)
        
        co_occ_dict[cat_i] = column
        df = pd.DataFrame(co_occ_dict, index=unique)

            
            
def embed(data, rca = True):
    df = prep_data(data)
    if rca:
        df = filter_df(df)
    co_occ_df = co_occurence(df)
    return co_occ_df


                


            


