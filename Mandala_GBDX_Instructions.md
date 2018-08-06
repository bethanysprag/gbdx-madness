# Order of Operations

1. Get GBDX token using gbdx user & password
~~~
curl -X POST \
  https://geobigdata.io/auth/v1/oauth/token/ \
  -H 'Authorization: Basic {{apikey}}' \
  -H 'Cache-Control: no-cache' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -H 'Postman-Token: 35124788-faa6-00f6-1c03-aaa48fc3fb17' \
  -d 'grant_type=password&username=<username>&password=<password>'
~~~

2. Get temporary AWS credentials
~~~
curl -X GET \
  'https://geobigdata.io/s3creds/v1/prefix?duration=3600' \
  -H 'Authorization: Bearer eyJ0eXAiOiJK...blah..blah' \
  -H 'Cache-Control: no-cache' \
  -H 'Postman-Token: ea5f5aea-d124-8840-8784-6abe198d1100'
~~~

3. Using the MegaWorkflow.json document as a template, make the follow changes:
  * Change both catalog IDs to match the images of interest
  * Change the output directory to the desired s3 folder.  GBDX will create a missing folder
  * Change the AWS credentials in the SaveToS3 section to use the temp credentials

4. POST the MegaWorkflow.json workflow
~~~
curl -X POST \
  https://geobigdata.io/workflows/v1/workflows \
  -H 'Authorization: Bearer eyJ0eXAiOiJK...blah..blah' \
  -H 'Cache-Control: no-cache' \
  -H 'Content-Type: application/json' \
  -H 'Postman-Token: 2649f23d-6f0d-a117-ee8b-56cfc92f2404' \
  -d '{<megaworkflow json code>}'
~~~

5. Record the workflow id that is returned from POSTING the previous JSON

6. Query the status of your workflow
~~~~
curl -X GET \
  https://geobigdata.io/workflows/v1/workflows/<workflow id> \
  -H 'Authorization: Bearer eyJ0eXAiOiJK...blah...blah' \
  -H 'Cache-Control: no-cache' \
  -H 'Content-Type: application/json' \
  -H 'Postman-Token: 07dc252f-b964-44a2-f008-c89d72c65b5a'
~~~

7. Configure AWS CLI to use your temp credentials
~~~
export AWS_ACCESS_KEY_ID=<access key>
export AWS_SECRET_ACCESS_KEY=<secret access key>
export AWS_SESSION_TOKEN=<session token>
~~~

8. Your data can be accessed in the output s3 bucket specified in step 3
~~~
aws s3 ls $OUTPUT_FOLDER
~~~
