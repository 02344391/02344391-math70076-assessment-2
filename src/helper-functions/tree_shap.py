

import numpy as np

def cat_tree_shap(x, nu, a, b, t, r, d):
    """
    """
    phi = np.zeros(len(x))
    def extend(m, pz, po, pi):
        """
        """
        l = len(m["weight"])
        m["feature"].append(pi)
        m["z"].append(pz)
        m["o"].append(po)
        m["weight"].append(int(l==0))
        for i in range(l-1,-1,-1):
            m["weight"][i+1] += po *  m["weight"][i] * (i+1)/(l+1)
            m["weight"][i] = pz * m["weight"][i] * (l-i)/(l+1)
        return m
    def unwind(m,i):
        """
        """
        l = len(m["weight"])
        n = m["weight"][l-1]
        o = m["o"][i]
        z = m["z"][i]
        for j in range(l-2,-1,-1):
            if o != 0:
                t = m["weight"][j]
                m["weight"][j] = n * (l) / ((j+1) * o)
                n = t - m["weight"][j] * z * ((l-1)-j)/l
            else:
                m["weight"][j] = (m["weight"][j] * l) / (z * (l-1-j))
        for j in range(i, l-1):
            m["feature"][j] = m["feature"][j+1]
            m["z"][j] = m["z"][j+1]
            m["o"][j] = m["o"][j+1]

        return m
    def sum_unwind(m, i):
        l = len(m["weight"])
        o = m["o"][i]
        z = m["z"][i]
        n = m["weight"][l-1]
        tot = 0
        for j in range(l-2, -1, -1):
            if o != 0:
                t = n * l/((j+1) * o)
                tot += t
                n = m["weight"][j] - t * z * (l-1-j)/(l)
                # print(f"if: {m['weight'][j]}, {z}, {l-1}, {n}, {t}")
            else:
                tot += (m["weight"][j] / z) / ((l - 1 - j) / l)
                # print(f"else: {m['weight'][j]}, {z}, {l-1}")
        return tot


    def findfirst(md, d_):
        try:
            k = np.where(md == d_)[0][0]
        except:
            k = None
        return k
    def recurse(j, dict_m, pz, po, pi):
        print(f"node {j}")

        """

        """
        m = dict_m[f"node {j}"].copy()
        m = extend(m, pz, po, pi)
        left = a[j]
        right = b[j]
        if right < 0:

            for i in range(1, len(m["weight"])):
                # print(m)
                w = sum_unwind(m, i)
                feature_index = int(m["feature"][i])
                phi[feature_index] = phi[feature_index] + w * (m["o"][i] - m["z"][i]) * nu[j]
        else:
            split = x[d[j]] <= t[j]
            h, c  = (left, right) * split + (right, left) * (1 - split)
            iz, io = 1, 1

            k = findfirst(np.array(m["feature"]), d[j])
            if k != None:
                iz, io = m["z"][k], m["o"][k]
                m = unwind(m,k)
            dict_m[f"node {h}"] = {"weight": [],
                                  "z": [],
                                  "o": [],
                                  "feature": []}
            dict_m[f"node {c}"] = {"weight": [],
                                  "z": [],
                                  "o": [],
                                  "feature": []}
            for key in dict_m[f"node {j}"]:
                dict_m[f"node {h}"][key] = m[key].copy()
                dict_m[f"node {c}"][key] = m[key].copy()
            print(dict_m)
            recurse(h, dict_m, iz * r[h]/r[j], io, d[j])
            recurse(c, dict_m, iz * r[c]/r[j], 0, d[j])
    recurse(0, {"node 0": {"weight": [],
                "z": [],
                "o": [],
                "feature": []}}, 1, 1, 0)
    return phi