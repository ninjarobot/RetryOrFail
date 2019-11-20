RetryOrFail
===========

_Determining whether an error is transient and a retry is likely to eventually succeed._

Error messages for recoverable errors are usually quite similar and a human will recognize that they can just retry the job. The goal here is to use ML.NET to make this determination automatically.

A machine learning model is trained based on error messages returned from an existing system (Ansible) the final status of whether the job eventually succeeded after one or more retries. The trained model is used to make predictions about new error messages and determine if the error is transient and will benefit from a retry or the job should just be failed.

Given some snippets from similar errors messages from jobs, the prediction engine is able to determine if a failure is transient and recoverable.

```
Likely to recover from 'foundation http timeout': True
Likely to recover from 'foundation password': False
Likely to recover from 'storage account not found': False
Likely to recover from 'error connecting': True
Likely to recover from 'something totally new': False
Likely to recover from 'there was an http timeout': True
Likely to recover from 'ssl connect timeout': True
Likely to recover from 'vm deploy': False
Likely to recover from 'http failure': True
```

