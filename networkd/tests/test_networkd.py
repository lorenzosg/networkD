from networkd import networkd as nd
embedder = nd.embed()

##unit tests for prep_data()

def test_prep_data_dict():
    '''
    make sure that the prep_data function can properly handle a dictionary as input 
    '''
    data = {'category': ['a', 'b', 'c'], 'entity': ['d', 'e', 'f'], 'value': [1,1,1]}
    expected_output = nd.pd.DataFrame({
        'category': ['a', 'b', 'c'], 
        'entity': ['d', 'e', 'f'], 
        'value': [1,1,1]
        })
    result = nd.prep_data(data)

    nd.pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_output.reset_index(drop=True))

test_prep_data_dict()

def test_prep_data_pd():
    '''
    make sure that the prep_data function can properly handle a pandas dataframe
    '''
    data = nd.pd.DataFrame({
        'category': ['a', 'b', 'c'], 
        'entity': ['d', 'e', 'f'], 
        'value': [1,1,1]})
    
    expected_output = nd.pd.DataFrame({
        'category': ['a', 'b', 'c'], 
        'entity': ['d', 'e', 'f'], 
        'value': [1,1,1]
        })
    result = nd.prep_data(data)

    nd.pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_output.reset_index(drop=True))

test_prep_data_pd()


def test_prep_data_no_third():
    '''
    make sure that the prep_data function can properly handle a pandas dataframe with only two columns
    '''
    data = nd.pd.DataFrame({
        'category': ['a', 'b', 'c'], 
        'entity': ['d', 'e', 'f'],
        })
    
    expected_output = nd.pd.DataFrame({
        'category': ['a', 'b', 'c'], 
        'entity': ['d', 'e', 'f'], 
        'value': [1,1,1]
        })
    result = nd.prep_data(data)

    nd.pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_output.reset_index(drop=True))

test_prep_data_no_third()


##unit tests for filter_df()

def test_filter_df():
    '''
    make sure that the rca is calculated properly 
    '''

    data = nd.pd.DataFrame({
        0: ['cat1', 'cat1', 'cat2', 'cat3'],
        1: ['ent1', 'ent2', 'ent1', 'ent3'],
        2: [2, 3, 4, 5]
    })

    expected_output = nd.pd.DataFrame({
        0: ['cat1', 'cat2', 'cat3'],
        1: ['ent2', 'ent1', 'ent3'],
        2: [3, 4, 5]
    })
    
    result = nd.filter_df(data)
    
    nd.pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_output.reset_index(drop=True))


test_filter_df()


def test_filter_df_empty_df():
    '''
    test to make sure that filtering an empty dataframe is handled gracefully 
    '''
    data = nd.pd.DataFrame(columns=[0, 1, 2])
    result = nd.filter_df(data)
    assert result.empty

test_filter_df_empty_df()



def test_filter_df_large_dataset():
    '''
    Test to make sure that rca can be calculated with a large dataset
    '''
    data = nd.pd.DataFrame({
        0: ['cat' + str(i % 100) for i in range(10000)],
        1: ['ent' + str(i % 100) for i in range(10000)],
        2: nd.np.random.randint(1, 100, 10000)
    })
    
    result = nd.filter_df(data)
    assert not result.empty

test_filter_df_large_dataset()


##unit tests for co_occurence()

def test_co_occurence_basic():
    '''
    Test to make sure that co-occurence works with self_loops. 
    '''
    data = nd.pd.DataFrame({
        0: ['cat1', 'cat1', 'cat2', 'cat3'],
        1: ['ent1', 'ent2', 'ent1', 'ent3']
    })
    result = nd.co_occurence(data, self_loops=True)
    
    expected_output = nd.pd.DataFrame({
        'cat1': [1, 0.5, 0],
        'cat2': [0.5, 1, 0],
        'cat3': [0, 0, 1]
    }, index=['cat1', 'cat2', 'cat3'])
    
    nd.pd.testing.assert_frame_equal(result, expected_output)

test_co_occurence_basic()

def test_co_occurence_no_self_loops():
    '''
    test to make sure that co-occurence works without self_loops
    '''

    data = nd.pd.DataFrame({
        0: ['cat1', 'cat1', 'cat2', 'cat3'],
        1: ['ent1', 'ent2', 'ent1', 'ent3']
    })
    result = nd.co_occurence(data, self_loops=False)
    
    expected_output = nd.pd.DataFrame({
        'cat1': [0, 0.5, 0],
        'cat2': [0.5, 0, 0],
        'cat3': [0, 0, 0]
    }, index=['cat1', 'cat2', 'cat3'])
    
    nd.pd.testing.assert_frame_equal(result, expected_output)

test_co_occurence_no_self_loops()


def test_co_occurence_empty_data():
    '''
    Test to make sure that an empty dataframe remains empty 
    '''
    data = nd.pd.DataFrame(columns=[0, 1])
    result = nd.co_occurence(data, self_loops=True)
    assert result.empty


test_co_occurence_empty_data()


##integration tests

data = nd.pd.DataFrame({
    0: ['cat1', 'cat1', 'cat2', 'cat3'],
    1: ['ent1', 'ent2', 'ent1', 'ent3'],
    2: [2, 3, 4, 5]
})          

def test_embed_basic():
    result = embedder.embed(data, rca=True, self_loops=True)
    
    expected_output = nd.pd.DataFrame({
        'cat1': [1, 0, 0],
        'cat2': [0, 1, 0],
        'cat3': [0, 0, 1]
    }, index=['cat1', 'cat2', 'cat3'])
    
    nd.pd.testing.assert_frame_equal(result, expected_output)

test_embed_basic()



def test_embed_rca_self_loops_false():
    '''
    integration test to verify expected output when rca = True
    and self_loops = False
    
    '''
    result = embedder.embed(data, rca=True, self_loops=False)
    
    expected_output = nd.pd.DataFrame({
        'cat1': [0, 0, 0],
        'cat2': [0, 0, 0],
        'cat3': [0, 0, 0]
    }, index=['cat1', 'cat2', 'cat3'])
    
    nd.pd.testing.assert_frame_equal(result, expected_output)

test_embed_rca_self_loops_false()