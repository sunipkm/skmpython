def staticvars(**kwargs):
    def decorate(func):
        for key in kwargs:
            setattr(func, key, kwargs[key])
        return func
    return decorate
