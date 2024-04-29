import numpy as np

def cat_tree_shap(x, nu, a, b, t, r, d):
    """
    """
    phi = np.zeros(len(x))
    def extend(m, pz, po, pi):
        """
        """
        l = len(m["weight"])
        weight = m["weight"].copy()
        feat = m["feature"].copy()
        z_m = m["z"].copy()
        o_m = m["o"].copy()
        m_copy = {"feature":feat,"z":z_m,"o":o_m,"weight":weight}
        m_copy["feature"].append(pi)
        m_copy["z"].append(pz)
        m_copy["o"].append(po)
        m_copy["weight"].append(int(l==0))
        print(m)
        for i in range(l-1,0,-1):
            # print(weight[i] + po *  weight[i-1] * (i/l))
            # print(m["weight"][i])
            weight = m["weight"].copy()
            m_copy["weight"][i] = weight[i] + po *  weight[i-1] * (i/l)
            m_copy["weight"][i-1] = pz * weight[i-1] * (l-i)/l
            # print(m_copy)
            print(m)

        return m_copy
    def unwind(m,i):
        """
        """
        l = len(m["weight"])
        n = m["weight"][l-1]
        m_cut = m.copy()
        for key in m_cut:
            m_cut[key] = m_cut[key][:-1]
        for j in range(l-1,0,-1):
            if m["o"][i] != 0:
                t = m["weight"][j-1]
                m_cut["weight"][j-1] = n * l / (j * m["o"][i])
                n = t - m_cut["weight"][j-1] * m["z"][i] * (l-j)/l
            else:
                m_cut["weight"][j-1] = (m_cut["weight"][j-1] * l) / (m["z"][i] * (l-j))
        for j in range(i, l-1):
            m_cut["feature"][j] = m["feature"][j+1]
            m_cut["z"][j] = m["z"][j+1]
            m_cut["o"][j] = m["o"][j+1]

        return m_cut
    def findfirst(md, d_):
        try:
            k = np.where(md == d_)[0][0]
        except:
            k = None
        return k
    def recurse(j, m, pz, po, pi):
        """

        """
        m = extend(m, pz, po, pi)
        internal = False
        left = a[j]
        right = b[j]
        if right > 0:
            internal = True
        if not internal:
            for i in range(1, len(m["weight"])):
                w = sum(unwind(m, i)["weight"])
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
            recurse(h, m, iz * r[h]/r[j], io, d[j])

            recurse(c, m, iz * r[c]/r[j], 0, d[j])
    recurse(0, {"feature": [],
                "z": [],
                "o": [],
                "weight": []}, 1, 1, 0)
    return phi