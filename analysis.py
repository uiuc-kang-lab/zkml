import json
import math
import matplotlib.pyplot as plt

"""
134367325.60817462
290775417.8
649149354.6
1537841530.4
3954863321.4
8532335876.1
18267801913.5
38864396676.6
82926802113.4
"""

my_list = []
normalize_list = []
for k in range(17, 25):
    with open('target/criterion/kzg_fft/k/{}/new/estimates.json'.format(k), 'r') as f:
        data = json.load(f)
        print(data['mean']['point_estimate'])
        my_list.append(data['mean']['point_estimate'])
        normalize_list.append(data['mean']['point_estimate'])


#plt.plot(range(18, 29), normalize_list)
#plt.xlabel('k')
#plt.ylabel('normalized time')
#plt.title('FFT')
#plt.show()
