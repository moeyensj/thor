# infra runbook #

## How to do a release:

Run the release script:

```
./release.sh
```

This will build a release based on the current Git version you have checked out.

You can specify a different Git ref with an argument to `./release.sh`:

```

./release.sh v1.2-rc0

./release.sh becd71868af

./release.sh fork/main
```
