plugins {
    id("java")
    application
}

application {
    mainClass.set("App")
}

group = "com.kadenfrisk"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.apache.commons:commons-csv:1.11.0")
    implementation("nz.ac.waikato.cms.weka:weka-stable:3.8.6")

    testImplementation(platform("org.junit:junit-bom:5.10.0"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}