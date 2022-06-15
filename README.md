docker build -t gcr.io/gnomondigital-sandbox/optimizer:0.3 .

docker run -ti --env GCLOUD_PROJECT=gnomondigital-sandbox -v=$HOME/.config/gcloud:/root/.config/gcloud gcr.io/gnomondigital-sandbox/optimizer:0.3 python app/main.py --env 1 --job_id 100004 --input_bucket gnomondigital-sandbox-tvm-job-input --output_bucket gnomondigital-sandbox-tvm-job-output --topic_id tvm-job-clean-topic --project_id gnomondigital-sandbox


gcloud compute instances create-with-container tvm-job-vm-100004 --project=gnomondigital-sandbox --zone=europe-west1-b --machine-type=n1-standard-1 --boot-disk-size=100G --container-image=gcr.io/gnomondigital-sandbox/optimizer:0.3 --service-account=tvm-job-runner@gnomondigital-sandbox.iam.gserviceaccount.com --scopes=cloud-platform  --container-command=python  --container-arg=main.py --container-arg='--path' --container-arg='gs://gnomondigital-sandbox-tvm-job-input/job_id=100004/config.json'
