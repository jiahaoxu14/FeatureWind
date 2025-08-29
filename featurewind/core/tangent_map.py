'''
Calculate the tangent map of a set of input data points using DimReader method 
Input: numeric and tabular dataset
Output the results in a JSON format.
'''
import numpy as np
import sys
from . import dim_reader as DimReader
import json

projections = DimReader.projections
projectionParamOpts = DimReader.projectionParamOpts
projectionClasses = DimReader.projectionClasses

def check_and_normalize_features(points):
    """
    Check if features are normalized to [0, 1] range, and normalize if needed.
    
    Args:
        points (list): List of data points, each point is a list of feature values
        
    Returns:
        list: Normalized points
    """
    if not points:
        return points
        
    points_array = np.array(points)
    n_points, n_features = points_array.shape
    
    print(f"Checking normalization for {n_features} features across {n_points} points...")
    
    # Check current feature ranges
    feature_mins = points_array.min(axis=0)
    feature_maxs = points_array.max(axis=0)
    
    print("Current feature ranges:")
    needs_normalization = False
    for i in range(n_features):
        print(f"  Feature {i}: [{feature_mins[i]:.3f}, {feature_maxs[i]:.3f}]")
        # Check if feature is not in [0, 1] range (with small tolerance)
        if feature_mins[i] < -0.001 or feature_maxs[i] > 1.001:
            needs_normalization = True
    
    if needs_normalization:
        print("Features are not normalized to [0, 1] range. Normalizing...")
        
        # Normalize each feature to [0, 1] range
        normalized_points = np.zeros_like(points_array)
        for i in range(n_features):
            feature_min = feature_mins[i]
            feature_max = feature_maxs[i]
            
            if feature_max > feature_min:  # Avoid division by zero
                normalized_points[:, i] = (points_array[:, i] - feature_min) / (feature_max - feature_min)
            else:
                # Constant feature - set to 0
                normalized_points[:, i] = 0.0
        
        print("Features normalized to [0, 1] range:")
        norm_mins = normalized_points.min(axis=0)
        norm_maxs = normalized_points.max(axis=0)
        for i in range(n_features):
            print(f"  Feature {i}: [{norm_mins[i]:.3f}, {norm_maxs[i]:.3f}]")
        
        # Convert back to list format
        return normalized_points.tolist()
    else:
        print("Features are already normalized to [0, 1] range.")
        return points

def calcTangentMap(points, projection, params):
    n, m = np.shape(points)  # n: number of data points, m: dimension of each data point
    tMap = []

    # Preparing tangent map structure
    for i in range(n):
        if isinstance(points[i], list):
            p = points[i]
        else:
            p = points[i].tolist()

        pt = {
            "domain": p,
            "range": [0, 0],
            "tangent": np.zeros((2, m)).tolist()
        }
        tMap.append(pt)
    print("Progress: 0%")

    # Projection runner initialization
    pj = DimReader.ProjectionRunner(projection, params)
    pj.firstRun = True
    pj.calculateValues(points)  # Initial run to get base projection
    base_proj = pj.outPoints  # Base projection points
    outPerts = pj.grads
    print(outPerts.shape)

    # Compute tangent map 
    pj.firstRun == False
    for i in range(m):
        for j in range(n):
            if isinstance(base_proj[j], list):
                tMap[j]["range"] = base_proj[j]
            else:
                tMap[j]["range"] = base_proj[j].tolist()
            
            tMap[j]["tangent"][0][i] = float(outPerts[j][0][i]) # derivative of x-coordinate
            tMap[j]["tangent"][1][i] = float(outPerts[j][1][i])  # derivative of y-coordinate

        print("tmap", tMap[i])
        print("Progress: ", int(((i + 1) / m) * 10000) / 100.0, "%")

    return tMap

if __name__ == "__main__":

    if (len(sys.argv) >= 3):
        inputFile = sys.argv[1]
        projection = sys.argv[2]

        if str.lower(projection) not in map(str.lower, DimReader.projections) and str.lower(projection) != "tangent-map":
            print("Invalid Projection")
            print("Projection Options:")
            for opt in DimReader.projections:
                if opt != "Tangent-Map":
                    print("\t" + opt)
            exit(0)

        projInd = list(map(str.lower, DimReader.projections)).index(str.lower(projection))

        inputPts = DimReader.readFile(inputFile)
        
        # Check if columns need normalization
        inputPts = check_and_normalize_features(inputPts)

        if (len(sys.argv)>3):
            params = []
            for i in range(3, len(sys.argv)):
                params = [sys.argv[i]]
        else:
            params = []

        tMap = calcTangentMap(inputPts,projectionClasses[projInd],params)

        fName = inputFile[:inputFile.rfind(".")]+"_TangentMap_"+projections[projInd] +".tmap"

        f = open(fName, "w")
        f.write(json.dumps(tMap))
        f.close()

    else:
        print("DimReaderScript [input file] [Projection] [optional parameters]")
        print("For all dimension perturbations, perturbation file = all")
        print("Projection Options:")
        for opt in projections:
            if opt!="Tangent-Map":
                print("\t" + opt)

        exit(0)