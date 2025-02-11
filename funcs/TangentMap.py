'''
Calculate the tangent map of a set of input data points using DimReader method 
Input: numeric and tabular dataset
Output the results in a JSON format.
'''
import numpy as np
import sys
import DimReader
import json

projections = DimReader.projections
projectionParamOpts = DimReader.projectionParamOpts
projectionClasses = DimReader.projectionClasses

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