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
     '''
     Intake a pandas dataframe with 3 columns and filter the values by 
     if the share of the category value within an entity is greater than 
     the share of the entire cateogry (data[0]) across all entities (data[1]). 
    
     Parameters
     ----------
     data: pandas dataframe 

     Returns
     -------
     filtered_data: adjacency matrix as an n(# of categories) by m(# of entities) numpy array. 
     '''
      
    cat_sums = data.groupby(data[0])[data[2]].sum()
    entity_sums = data.groupby(data[1])[data[2]].sum()
    total_sums = data[2].sum()

    data['rca_num'] = data[2] / data[1].map(entity_sums)
    data['rca_denom'] = data[0].map(cat_sums) / total_sums
    data['rca'] = data['rca_num'] / data['rca_denom'] 

    filtered_data = data[data['rca'] >= 1].drop(columns = ['rca_num', 'rca_denom', 'rca'])

    return filtered_data

            

def co_occurence(data):
     '''
     Intake an rca filtered pandas dataframe and calculate a co-occurence matrix by 
     computing the conditional probability that given the most frequent category another 
     category will also appear in the same entity for all pairings of categories. 

     Parameters
     ----------
     data: pandas dataframe 

     Returns
     -------
     df: pandas dataframe
    
     '''
  
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

    return df 
            
            
def embed(data, rca = True):
    '''
     Call helper functions prep_data and filter_df if necessary in order to embed the data
     by constructing a co-occurence matrix of the bi-partite graph.  
    
     Parameters
     ----------
     data: pandas dataframe 

     Returns
     -------
     co_occ_df: pandas dataframe of the co-occurence matrix. 
     '''
    df = prep_data(data)
    if rca:
        df = filter_df(df)
    co_occ_df = co_occurence(df)
    return co_occ_df


                


            


