import pandas as pd
import numpy as np


def prep_data(self, data):
     '''
     Intake a pandas dataframe or dictionary of two series or lists of type categorical which 
     describe the occurence of categories(1st) inside the entities (2nd) of a bi-parite graph. 
     A 3rd numerical column which describes the degree of the relationship of the category with 
     the entity is optional. 
    
     Parameters
     ----------
     bipartite: pandas dataframe or dicionary of two series or lists of type categorical

     Returns
     -------
     adj_df: adjacency matrix as an n(# of categories) by m(# of entities) numpy array. 
     '''

     if isinstance(data, dict):
         data = pd.DataFrame(data)
     else:
         raise TypeError(f'data must be a pandas dataframe or dictionary. Got {type(data)}')


     if len(data.columns) < 3:
         data['value'] = 1
         col_names = data.columns 
         adj_df = data.pivot(index = col_names[0], columns = col_names[1], values = col_names[2])
         adj_mat = adj_df.to_numpy()

     return adj_mat
     
 
        







def build_network(self, adj_df):
        '''
        Intake pandas df of adjacency matrix from image classifier and outputs symetric normalized covariance matrix of                 clothing items 
        elements at index [i,j] are the conditional probability of two clothing items being styled together in the same image 
        
        Parameters
        ----------
        adj_df: pandas dataframe of adjacency matrix of how often clothing items are styled in X posts 

        Returns
        -------
        network: symetric normalized covariance matrix of clothing items elements at index [i,j] 
        are the conditional probability of two clothing items being styled together in the same image 
        pieces: a list of the article descriptions from column names of adj_df


        '''
        
        
        pieces = list(adj_df.index)
        adj_m = adj_df.to_numpy()
        
        

        adj_m = adj_m
        adj_m_t = adj_m.transpose()
        
        

        mat_dot = np.dot(adj_m, adj_m_t)
        num_photos = np.count_nonzero(adj_m, axis=1) #get number of photos which the article of clothing is styled in


        #normalize matrix
        network = mat_dot / num_photos 
        network2 = mat_dot / num_photos

        #fill diagonals with zeros so that the network doesn't have any self-loops
        np.fill_diagonal(network, 0)

        #get the indices for the upper and lower traingles of the matrix
        i_lower = np.tril_indices(len(mat_dot))
        i_upper = np.triu_indices(len(mat_dot))

        #make each matrix symmetrical 
        network[i_lower] = network.T[i_lower]
        network2[i_upper] = network2.T[i_upper]

        #We want to conservatively estimate the similarity by taking the lower value
        
        for i in range(len(network)):
            for j in range(len(network)):
                if network2[i][j] < network[i][j]:
                    network[i][j] = network2[i][j]
                else:
                    pass

        return network, pieces
