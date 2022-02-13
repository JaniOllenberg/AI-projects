E=2.71828182846

layer_output = [1,2,3,4,5]

exponentiated_values = []
for value in layer_output:
    exponentiated_values.append(E**value)

norm_base = sum(exponentiated_values)
norm_values = []
for value in exponentiated_values:
    norm_values.append(value/norm_base)

print(norm_values)
print(sum(norm_values))
