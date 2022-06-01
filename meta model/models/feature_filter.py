#region imports

#endregion

#region base class
class SALA_FeatureFilter:
    name = ""

    def __init__(self, name = "not set"):
        self.name = name

    def run(self):
        pass
#endregion

#region filters
class SALA_ListFilter(SALA_FeatureFilter):
    def run(self, df):
        droppable = ['ELNRisk']

        targets = []
        for feature in df.columns:
            for drop in droppable:
                if str(feature).startswith(drop):
                    targets.append(str(feature))

        df = df.drop(columns=targets)

        return df
#endregion