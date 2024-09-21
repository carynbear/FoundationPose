```
  cd docker/
  #build from scratch
  docker build --network host -t foundationpose:live .
  bash docker/run_container.sh
```

If it's the first time you launch the container, you need to build extensions.
```
bash build_all.sh
```

Later you can execute into the container without re-build.
```
docker exec -it foundationpose bash
```