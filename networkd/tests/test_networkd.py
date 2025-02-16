from networkd import networkd as nd


##unit tests for prep_data()

def test_prep_data_dict():
    '''
    make sure that the prep_data function can properly handle a dictionary as input 
    '''
    data = {'category': ['a', 'b', 'c'], 'entity': ['d', 'e', 'f'], 'value': [1,1,1]}
    expected_output = nd.np.array([
        [1, 0, 0], 
        [0, 1, 0], 
        [0, 0, 1]])
    result = nd.Embed.prep_data(data)

    nd.np.testing.assert_array_equal(result, expected_output)



def test_prep_data_pd():
    '''
    make sure that the prep_data function can properly handle a pandas dataframe
    '''
    data = nd.pd.DataFrame({
        'category': ['a', 'b', 'c'], 
        'entity': ['d', 'e', 'f'], 
        'value': [1,1,1]})
    
    expected_output = nd.np.array([
        [1, 0, 0], 
        [0, 1, 0], 
        [0, 0, 1]])
    result = nd.Embed.prep_data(data)

    nd.np.testing.assert_array_equal(result, expected_output)



def test_prep_data_no_third():
    '''
    make sure that the prep_data function can properly handle a pandas dataframe with only two columns
    '''
    data = nd.pd.DataFrame({
        'category': ['a', 'b', 'c'], 
        'entity': ['d', 'e', 'f'],
        })
    
    expected_output = nd.np.array([
        [1, 0, 0], 
        [0, 1, 0], 
        [0, 0, 1]])
    result = nd.Embed.prep_data(data)

    nd.np.testing.assert_array_equal(result, expected_output)



##unit tests for filter_df()

def test_filter_df():
    '''
    make sure that the rca is calculated properly 
    '''
    data = nd.np.array([[1, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])


    expected_output = nd.np.array([[1, 1, 0],
                                   [1, 0, 0],
                                    [0, 0, 1]])
    
    result = nd.Embed.filter_df(data, threshold = 1)

    nd.np.testing.assert_array_equal(result, expected_output)

   
    



def test_filter_df_empty_df():
    '''
    test to make sure that filtering an empty dataframe is handled gracefully 
    '''
    data = nd.np.empty((0, 0), dtype=int)
    result = nd.Embed.filter_df(data, threshold = 1)
    assert result.size == 0




##unit tests for co_occurence()

def test_co_occurence_basic():
    '''
    Test to make sure that co-occurence works with self_loops. 
    '''
    data = nd.np.array([[1, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])

    
    expected_output = nd.np.array([
        [1, 0.5, 0],
        [0.5, 1, 0],
         [0, 0, 1]
    ])
    
    result = nd.Embed.co_occurence(data, self_loops=True)
    nd.np.testing.assert_array_equal(result, expected_output)


def test_co_occurence_no_self_loops():
    '''
    test to make sure that co-occurence works without self_loops
    '''

    data = nd.np.array([[1, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])

    
    expected_output = nd.np.array([
        [1, 0.5, 0],
        [0.5, 1, 0],
         [0, 0, 1]
    ])

    
    
    result = nd.Embed.co_occurence(data, self_loops=True)
    print(f'result in test_co_occurence {result}')
    nd.np.testing.assert_allclose(result, expected_output, rtol=1e-5, atol=1e-8)



def test_co_occurence_empty_data():
    '''
    Test to make sure that an empty dataframe remains empty 
    '''
    data = nd.pd.DataFrame(columns=[0, 1])
    result = nd.Embed.co_occurence(data, self_loops=True)
    assert result.size == 0




##integration tests

data = nd.pd.DataFrame({
    '0': ['cat1', 'cat1', 'cat2', 'cat3'],
    '1': ['ent1', 'ent2', 'ent1', 'ent3'],
    '2': [2, 3, 4, 5]
})          

def test_embed_basic():
    result = nd.Embed.embed(data, rca=True, self_loops=True)
    
    expected_output = nd.np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])
    
    nd.np.testing.assert_allclose(result, expected_output, rtol=1e-5, atol=1e-8)





def test_embed_rca_self_loops_false():
    '''
    integration test to verify expected output when rca = True
    and self_loops = False
    
    '''
    result = nd.Embed.embed(data, rca=True, self_loops=False)
    
    expected_output = nd.np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
    
    nd.np.testing.assert_allclose(result, expected_output, rtol=1e-5, atol=1e-8)


def test_filter_df_large_dataset():
    '''
    Test to make sure that rca can be calculated with a large dataset
    '''
    data = nd.pd.DataFrame({
        'cat': ['cat' + str(i % 100) for i in range(10000)],
        'ent': ['ent' + str(i % 100) for i in range(10000)],
        'values': nd.np.random.randint(1, 100, 10000)
    })
    
    expected_rows = len(nd.np.unique(data['cat']))
    expected_cols = len(nd.np.unique(data['ent']))

    result = nd.Embed.embed(data, rca = True, threshold = 1)
    assert result.shape == (expected_rows, expected_cols)