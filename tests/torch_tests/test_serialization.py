import torch

def test_save_load():
    state_dict = {
        'a.test.weight': torch.randn(3,4),
        'b.test.weight': torch.randn(4,5)
    }
    
    print(state_dict['a.test.weight'].__reduce_ex__)
    torch.save(state_dict, 'test.pt')
    state_dict_load = torch.load('test.pt')
    for k, v in state_dict.items():
        print(type(v), type(state_dict_load[k]))
        print(v, state_dict_load[k])
        assert torch.equal(v, state_dict_load[k])

def test_pickle():
    import pickle

    obj = torch.Tensor(42)
    serialized = pickle.dumps(obj)
    print(serialized)

    deserialized = pickle.loads(serialized)
    print(deserialized.value)  # 输出: 42
