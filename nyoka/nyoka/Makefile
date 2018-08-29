/nyoka/PMML43.py:
	mkdir -p nyoka
	mkdir -p nyoka/pmml
	make -C $(CURDIR)/PMML43/
	make -C $(CURDIR)/PMML43Ext/
	cp $(CURDIR)/PMML43/pmml43.py $(CURDIR)/nyoka/pmml/PMML43Super.py
	cp $(CURDIR)/PMML43/nyoka_local.py $(CURDIR)/nyoka/pmml/PMML43.py
	cp $(CURDIR)/PMML43/Base64.py $(CURDIR)/nyoka/pmml/Base64.py
	cp $(CURDIR)/PMML43Ext/pmml43Ext.py $(CURDIR)/nyoka/pmml/PMML43ExtSuper.py
	cp $(CURDIR)/PMML43Ext/nyoka_local.py $(CURDIR)/nyoka/pmml/PMML43Ext.py
	sed -i '' s+pmml43+nyoka.pmml.PMML43Super+g $(CURDIR)/nyoka/pmml/PMML43.py
	sed -i '' s+pmml43Ext+nyoka.pmml.PMML43ExtSuper+g $(CURDIR)/nyoka/pmml/PMML43Ext.py
	sed -i '' -E 's+def parse\(+def parseSub\(+g' $(CURDIR)/nyoka/pmml/PMML43.py
	sed -i '' -E 's+def parse\(+def parseSub\(+g' $(CURDIR)/nyoka/pmml/PMML43Ext.py
	cat wrapper43.py >> nyoka/pmml/PMML43.py
	cat wrapper43Ext.py >> nyoka/pmml/PMML43Ext.py

.PHONY: clean test doc
clean:
	make -C $(CURDIR)/PMML43/ clean
	make -C $(CURDIR)/PMML43Ext/ clean
	make -C $(CURDIR)/doc/ clean
	-rm *.pyc
	-rm nyoka/*.pyc
	-rm nyoka/pmml/*.pyc
	-rm -rf nyoka/__pycache__
	-rm -rf nyoka/pmml/__pycache__

test:
	pip3 install --user numpy
	pip3 install --user jupyter
	python3 test_script.py script_test.pmml script_tested.pmml
	python3 test2.py
	jupyter nbconvert --execute Nyoka_String_Export.ipynb
	jupyter nbconvert --execute Nyoka_Script_Tag.ipynb

doc:
	./build.py
	pip install sphinx --user
	make -C $(CURDIR)/doc/ html
	-open file://$(CURDIR)/doc/build/html/index.html
	-xdg-open file://$(CURDIR)/doc/build/html/index.html
