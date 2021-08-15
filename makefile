jpf.jar:
	javac jpf/*.java
	jar -cf jpf.jar jpf/*.java

tests: jpf.jar
	javac Tests.java
	
examples: jpf.jar
	javac Examples.java

clean:
	rm jpf/*.class
	rm *.class