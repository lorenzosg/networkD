from networkd import networkd as nd


def test_prep_data():
    example = {'category': ['a', 'b', 'c'], 'entity': ['d', 'e', 'f'], 'value': [1,0,1]}
    expected = nd.pd.DataFrame
    actual = type(nd.prep_data(example))
    assert actual == expected 

test_prep_data()


def test_filter_df():
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
    data = nd.pd.DataFrame(columns=[0, 1, 2])
    result = nd.filter_df(data)
    assert result.empty

test_filter_df_empty_df()



def test_filter_df_large_dataset():
    data = nd.pd.DataFrame({
        0: ['cat' + str(i % 100) for i in range(10000)],
        1: ['ent' + str(i % 100) for i in range(10000)],
        2: nd.np.random.randint(1, 100, 10000)
    })
    
    result = nd.filter_df(data)
    assert not result.empty

test_filter_df_large_dataset()



def test_co_occurence_basic():
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
    data = nd.pd.DataFrame(columns=[0, 1])
    result = nd.co_occurence(data, self_loops=True)
    assert result.empty


test_co_occurence_empty_data()


