FROM openjdk:17-jdk

WORKDIR /MDM_Project2

COPY  target/demo-0.0.1-SNAPSHOT.jar mdm_project.jar

EXPOSE 8081

CMD ["java", "-jar", "/MDM_Project2/mdm_project.jar"]
