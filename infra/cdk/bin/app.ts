#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { WebStack } from "../lib/web-stack";

const app = new App();

const environmentName = app.node.tryGetContext("environmentName") ?? "dev";

new WebStack(app, `TraductorKaqchikel-Web-${environmentName}`, {
  environmentName,
});
