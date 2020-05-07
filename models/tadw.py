import networkx as nx
import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import normalize

from models.base_model import BaseModel


class TADW(BaseModel):
    def __init__(self, graph, features, dim=80, lamb=0.2, **kwargs):
        super(TADW, self).__init__(graph, features, dim, **kwargs)

        self.g = graph
        self.features = np.array(features)
        self.lamb = lamb
        self.dim = int(dim/2)

        self.embeddings = None

    def get_adj(self):
        adj = nx.to_numpy_matrix(self.g).A
        # ScaleSimMat
        return adj/np.sum(adj, axis=1)

    def get_embeddings(self):
        return self.embeddings

    def get_t(self):
        self.preprocess_feature()
        return self.features.T

    def preprocess_feature(self):
        if self.features.shape[1] > 200:
            U, S, VT = la.svd(self.features)
            Ud = U[:, 0:200]
            Sd = S[0:200]
            self.features = np.array(Ud)*Sd.reshape(200)

    def learn_embeddings(self):
        self.adj = self.get_adj()
        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix
        self.M = (self.adj + np.dot(self.adj, self.adj))/2
        # T is feature_size*node_num, text features
        self.T = self.get_t()
        self.node_size = self.adj.shape[0]
        self.feature_size = self.features.shape[1]
        self.W = np.random.randn(self.dim, self.node_size)
        self.H = np.random.randn(self.dim, self.feature_size)
        # Update
        for i in range(20):
            print('Iteration ', i)
            # Update W
            B = np.dot(self.H, self.T)
            drv = 2 * np.dot(np.dot(B, B.T), self.W) - \
                2*np.dot(B, self.M.T) + self.lamb*self.W
            Hess = 2*np.dot(B, B.T) + self.lamb*np.eye(self.dim)
            drv = np.reshape(drv, [self.dim*self.node_size, 1])
            rt = -drv
            dt = rt
            vecW = np.reshape(self.W, [self.dim*self.node_size, 1])
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.node_size))
                Hdt = np.reshape(np.dot(Hess, dtS), [
                                 self.dim*self.node_size, 1])

                at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
                vecW = vecW + at*dt
                rtmp = rt
                rt = rt - at*Hdt
                bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.W = np.reshape(vecW, (self.dim, self.node_size))

            # Update H
            drv = np.dot((np.dot(np.dot(np.dot(self.W, self.W.T), self.H), self.T)
                          - np.dot(self.W, self.M.T)), self.T.T) + self.lamb*self.H
            drv = np.reshape(drv, (self.dim*self.feature_size, 1))
            rt = -drv
            dt = rt
            vecH = np.reshape(self.H, (self.dim*self.feature_size, 1))
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.feature_size))
                Hdt = np.reshape(np.dot(np.dot(np.dot(self.W, self.W.T), dtS), np.dot(self.T, self.T.T))
                                 + self.lamb*dtS, (self.dim*self.feature_size, 1))
                at = np.dot(rt.T, rt)/np.dot(dt.T, Hdt)
                vecH = vecH + at*dt
                rtmp = rt
                rt = rt - at*Hdt
                bt = np.dot(rt.T, rt)/np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.H = np.reshape(vecH, (self.dim, self.feature_size))

        self.embeddings = np.hstack((normalize(self.W.T), normalize(np.dot(self.T.T, self.H.T))))
