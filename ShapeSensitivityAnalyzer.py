import polars as pl
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class ShapeSensitivityAnalyzer:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        
    def z_scale_all_data(self):
        # Here, we want to find the z-score for all the data, and return a dataframe'
        # that uses this more manageable scale
        
        # Polars gives us convinient ways to find the mean and standard deviation
        self.scaled_df = self.df
        # skip over the first 4 columns, because these are not data, but rotations and delta 
        for idx, colname in enumerate(self.df.columns[4:]):
            df_idx = idx + 4
            mean = self.df[colname].mean()
            std = self.df[colname].std()
            col_len = self.df[colname].shape[0]
            # check if stddev is zero (if so, we can't scale)
            if std == 0:
                # set the z-score to array of zeros 
                z_score = pl.zeros(col_len, eager=True).to_numpy()
            else:
                z_score = (self.df[colname].to_numpy() - mean) / std
            self.scaled_df.replace(colname, pl.Series(colname, z_score))

        
    
    def row_norms_per_rotation_nudge(self):
        # we are looking at the sensitivity of the shape to each rotation
        # want to extract out the rotations only for specidfiz dx,dy,dz nudges
        
        normed_df = pl.DataFrame()
        # start with dx
        dx_df = self.scaled_df.filter(pl.col("delta") == "dx")
        # find the norm of each row
        dx_norm = dx_df[:, 4:].map_rows(lambda row: np.linalg.norm(row))
        dx_norm = dx_norm.rename({"map" : "dx_norm"})
        
        # now dy
        dy_df = self.scaled_df.filter(pl.col("delta") == "dy")
        dy_norm = dy_df[:, 4:].map_rows(lambda row: np.linalg.norm(row))
        dy_norm = dy_norm.rename({"map" : "dy_norm"})
        # now dz
        dz_df = self.scaled_df.filter(pl.col("delta") == "dz")
        dz_norm = dz_df[:, 4:].map_rows(lambda row: np.linalg.norm(row))
        dz_norm = dz_norm.rename({"map" : "dz_norm"})
        # concatenate norms, and have input x,y,z values from orignal dataframe
        normed_df = pl.concat(
            [dx_df[:,0:3], dx_norm, dy_norm, dz_norm], 
            how = "horizontal"
        )
        # rename columns
        self.normed_df = normed_df
    def find_min_max_regions(self, colname: str):
        min_range = self.df[colname].min() +  abs((self.df[colname].max() -  self.df[colname].min())*0.05)
        
        return self.df.filter(
            (pl.col(colname) >= self.df[colname].min()) & (pl.col(colname) <=  min_range)
        )
    def cluster_by_sensitivity(self, num_clusters: int = 3):
        # We want to cluster by the sensitivity of the shape
        # But we care about the distribution of x,y,z rotations (e,e - rows of df)
        
        km = KMeans(n_clusters=num_clusters)
        # apply cluster only to sensitivity
        km.fit(self.normed_df[:, 4:].to_numpy())
        # grab the cluster labels
        cluster_labels = km.labels_
        # add the cluster labels to the df
        self.normed_df = self.normed_df.with_columns(pl.Series("cluster", cluster_labels))
        
    def cluster_by_sensitivity_positive_rots(self, num_clusters: int = 3):
        km = KMeans(n_clusters=num_clusters)
        # apply cluster only to sensitivity
        # filter out negative rotations 
        pos_df = self.df.filter(
            (pl.col("x") >= 0) & (pl.col("y") >= 0) & (pl.col("z") >= 0)
            )
        km.fit(pos_df[:, 3:].to_numpy())
        # grab the cluster labels
        cluster_labels = km.labels_
        # add the cluster labels to a modified, filtered, df
        self.pos_df = pos_df.with_column(pl.Series("cluster", cluster_labels))
    
    def average_out_z_rotations(self):
        self.z_avg_df = self.normed_df.group_by(
            ["x", "y"]
        ).agg([
            pl.mean("dx_norm"),
            pl.mean("dy_norm"),
            pl.mean("dz_norm")
        ]
        )
    def plot_sensitivity_surfaces(self, max_z=30):
        x_data = self.z_avg_df["x"].to_numpy()
        y_data = self.z_avg_df["y"].to_numpy()
        dx_height = self.z_avg_df['dx_norm'].to_numpy()
        dy_height = self.z_avg_df['dy_norm'].to_numpy()
        dz_height = self.z_avg_df['dz_norm'].to_numpy()

        fig_dx = plt.figure(figsize=(10,10))
        ax_dx = fig_dx.add_subplot(projection='3d')
        ax_dx.set_xlabel('x')
        ax_dx.set_ylabel('y')
        ax_dx.set_zlabel('dx')
        # set z-range from 0-30
        ax_dx.set_zlim(0, max_z)


        # want to do the same thing for dy
        fig_dy = plt.figure(figsize=(10,10))
        ax_dy = fig_dy.add_subplot(projection='3d')
        ax_dy.set_xlabel('x')
        ax_dy.set_ylabel('y')
        ax_dy.set_zlabel('dy')
        ax_dy.set_zlim(0, max_z)


        # want to do the same thing for dz
        fig_dz = plt.figure(figsize=(10,10))
        ax_dz = fig_dz.add_subplot(projection='3d')
        ax_dz.set_xlabel('x')
        ax_dz.set_ylabel('y')
        ax_dz.set_zlabel('dz')
        ax_dz.set_zlim(0, max_z)

        surfdx = ax_dx.plot_trisurf(x_data, y_data, dx_height, cmap='viridis', edgecolor='none')
        surfdy = ax_dy.plot_trisurf(x_data, y_data, dy_height, cmap='viridis', edgecolor='none')
        surfdz = ax_dz.plot_trisurf(x_data, y_data, dz_height, cmap='viridis', edgecolor='none')
        
        
        