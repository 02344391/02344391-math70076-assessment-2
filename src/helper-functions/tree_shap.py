import numpy as np

def cat_tree_shap(x, nu, a, b, t, r, d):
    """
    """
    phi = np.zeros(len(x))
    def extend(m, pz, po, pi):
        """
        """
        print(m)
        l, m = len(m), m.copy()
        m.append(np.array([pi, pz, po, int(l==0)]))
        for i in range(l-1,-1,-1):
            m[i+1][3] = m[i+1][3] + po * m[i][3] * ((i+1)/l)
            m[i][3] = pz * m[i][3] * (l-(i+1))/l
        return m
    def unwind(m,i):
        """
        """
        l = len(m)
        n = m[l-1][3]
        m_cut = m[:l-1].copy()
        for j in range(l-2,-1,-1):
            if m[i][2] != 0:
                t = m[j][3]
                m_cut[j][3] = n * l / ((j+1) * m[i][2])
                n = t - m_cut[j][3] * m[i][1] * (l-(j+1))/l
            else:
                m_cut[j][3] = (m_cut[j][3] * l) / (m[i][1] * (l-(j+1)))
        for j in range(i, l-1):
            m_cut[j][:-1] = m[j+1][:-1]
        return m_cut
    def findfirst(md, d):
        try:
            k = np.where(md == d)[0][0]
        except:
            k = None
        return k
    def recurse(j, m, pz, po, pi):
        """

        """
        m = extend(m, pz, po, pi)
        internal = False
        try:
            left = a[j]
            right = b[j]
            if left > 0:
                internal = True
        except:
            pass
        if not internal:
            for i in range(1, len(m)):
                undo = unwind(m, i)
                w = sum([node[3] for node in undo])
                feature_index = int(m[i][0])
                phi[feature_index] = phi[feature_index] + w * (m[i][2] - m[i][1]) * nu[j]
        else:
            split = x[d[j]] <= t[j]
            h, c = (left, right) * split + (right, left) * (1 - split)
            iz, io = 1, 1
            k = findfirst(np.array([node[0] for node in m]), d[j])
            if k != None:
                iz, io = m[k][1], m[k][2]
                m = unwind(m,k)
                print(m)
            recurse(h, m, iz * r[h]/r[j], io, d[j])

            recurse(c, m, iz * r[c]/r[j], 0, d[j])
    recurse(0, [], 1, 1, 0)
    return phi