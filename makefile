.PHONY: build
build:
	fpm build

.PHONY: run
run:
	fpm run

.PHONY: test
test:
	@if [ "$(MAKECMDGOALS)" = "test" ]; then \
		fpm test; \
	else \
		for goal in $(MAKECMDGOALS); do \
			if [ $$goal != "test" ]; then \
				fpm test $$goal; \
			fi; \
		done; \
	fi

.PHONY: doc
doc:
	ford Fortran_Neural_Network.md -d src/ -o src/doc
	echo "#!/bin/bash" > src/doc/server.sh
	echo "set -xe" >> src/doc/server.sh
	echo "python -m http.server 9000 -d src/doc" >> src/doc/server.sh
	chmod +x src/doc/server.sh
	alacritty -e src/doc/server.sh &
	firefox localhost:9000

.PHONY: clean
clean:
	fpm clean --all --registry-cache
	rm -rf src/doc
