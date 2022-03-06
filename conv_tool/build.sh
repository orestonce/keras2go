set -xe

rm -f *.go
python -m keras2go --num_tests 3 --model_path ./model.h5 --function_name Example --package_name example
go fmt *.go
go test -v .
