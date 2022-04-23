# TODO: 研究这是干啥的
class ClassMining(object):
    @classmethod
    def getSubclassList(cls, model):
        subclass_list = []
        for subclass in model.__subclasses__():
            subclass_list.append(subclass)
            subclass_list.extend(cls.getSubclassList(subclass))
        return subclass_list

    @classmethod
    def getSubclassDict(cls, model):
        subclass_list = cls.getSubclassList(model=model)
        return {k: k.__name__ for k in subclass_list}

    @classmethod
    def getSubclassNames(cls, model):
        subclass_list = cls.getSubclassList(model=model)
        return [k.__name__ for k in subclass_list]

    @classmethod
    def getInstanceBySubclassName(cls, model, name):
        for subclass in model.__subclasses__():
            if subclass.__name__ == name:
                return subclass
            instance = cls.getInstanceBySubclassName(subclass, name)
            if instance:
                return instance
