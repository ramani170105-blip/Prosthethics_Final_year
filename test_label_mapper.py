from fusion.label_mapper import LabelMapper

mapper = LabelMapper()

print(mapper.map_physionet(3, "T1"))
print(mapper.map_physionet(3, "T2"))
print(mapper.map_physionet(5, "T1"))
print(mapper.map_physionet(6, "T2"))