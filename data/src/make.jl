# Convert the original example datasets to a compressed CSV file

# See data/README.md for the source of the input data files
# To regenerate the output file:
# 1) Have all input files ready in the data folder
# 2) Instantiate the package environment for data/src
# 3) Run this script with the root folder of the repository being the working directory

using CSV, CodecZlib, DataFrames, ReadStatTables

function stata_examples()
    fnames = ["docvisits", "poisson1"]
    for f in fnames
        df = DataFrame(readstat("data/$f.dta"))
        open(GzipCompressorStream, "data/$f.csv.gz", "w") do stream
            CSV.write(stream, df)
        end
    end
end

function main()
    stata_examples()
end

main()
