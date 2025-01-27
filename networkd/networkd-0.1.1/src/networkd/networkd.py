import pandas as pd
import numpy as np

class Embed: 

    @staticmethod
    def prep_data(data):
        '''
        Intake a pandas dataframe or dictionary of two series or lists of type categorical which 
        describe the occurence of categories(1st) inside the entities (2nd) of a bi-parite graph. 
        A 3rd numerical column which describes the degree of the relationship of the category within 
        the entity is optional. If not inlcuded a column of 1's will be assigned. 
        
        Parameters
        ----------
        data: pandas dataframe or dicionary of two series or lists of type categorical

        Returns
        -------
        adj_df: pandas dataframe with 3 columns, category, entity, value, in that order.  
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
        
        return data
        
    @staticmethod
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

        col_names = data.columns
        cat_sums = data.groupby(data[col_names[0]])[col_names[2]].sum()
        entity_sums = data.groupby(data[col_names[1]])[col_names[2]].sum()
        total_sums = data[col_names[2]].sum()

        data['rca_num'] = data[col_names[2]] / data[col_names[1]].map(entity_sums)
        data['rca_denom'] = data[col_names[0]].map(cat_sums) / total_sums
        data['rca'] = data['rca_num'] / data['rca_denom'] 

        filtered_data = data[data['rca'] >= 1].drop(columns = ['rca_num', 'rca_denom', 'rca'])

        for col in list(data.columns)[:2]:
            filtered_data[col] = filtered_data[col].astype(data[col].dtype)
        
        filtered_data = filtered_data.reset_index(drop = True)
        
        return filtered_data

                
    @staticmethod
    def co_occurence(data, self_loops):
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
        col_names = data.columns
        co_occ_dict = {}
        unique = sorted(set(data[col_names[0]]))
        for cat_i in unique:
            entities_i = set(data[col_names[1]][data[col_names[0]] == cat_i])
            column = []
            for cat_j in unique:
                if cat_i == cat_j:
                    if self_loops:
                        column.append(1)
                    else:
                        column.append(0)
                else:
                    entities_j = set(data[col_names[1]][data[col_names[0]] == cat_j])
                    shared = entities_i & entities_j
                    cond_prob = len(shared) / max(len(entities_i), len(entities_j))
                    column.append(cond_prob)
            
            co_occ_dict[cat_i] = column
        
        co_occ_df = pd.DataFrame(co_occ_dict, index=unique)
        return co_occ_df 
                
    @staticmethod            
    def embed(data, rca = False, self_loops = True):
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
        df = Embed.prep_data(data)
        if rca:
            df = Embed.filter_df(df)
        co_occ_df = Embed.co_occurence(df, self_loops)
        return co_occ_df




            


