import json
from pathlib import Path

import argschema

from deepinterpolation.cli.schemas import FineTuningInputSchema
from deepinterpolation.generic import ClassLoader


class FineTuning(argschema.ArgSchemaParser):
    default_schema = FineTuningInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        uid = self.args["run_uid"]

        outdir = Path(self.args["finetuning_params"]["output_dir"])
        if self.args["output_full_args"]:
            full_args_path = outdir / f"{uid}_training_full_args.json"
            with open(full_args_path, "w") as f:
                json.dump(self.args, f, indent=2)
            self.logger.info(f"wrote {full_args_path}")

        # We create the output model filename if empty
        if self.args["finetuning_params"]["model_string"] == "":
            self.args["finetuning_params"]["model_string"] = self.args[
                "finetuning_params"
            ]["loss"]

        # TODO: The following lines will be remove once we deprecate the legacy
        # parameter tracking system

        # We pass on the uid
        self.args["finetuning_params"]["run_uid"] = uid

        # We convert to old schema
        self.args["finetuning_params"]["nb_gpus"] = 2 * int(
            self.args["finetuning_params"]["multi_gpus"]
        )
        ## data shape, facilitate training on multiple GPUs
        if 'field_ids' in self.args["generator_params"]:
            self.args["generator_params"]["field_ids"] = self.args["generator_params"]["field_ids"]
        if 'channel_id' in self.args["generator_params"]:
            self.args["generator_params"]["channel_ids"] = self.args["generator_params"]["channel_id"]
        if 'image_hight' in self.args["generator_params"]:
            self.args["generator_params"]["image_hight"] = self.args["generator_params"]["image_hight"]
        if 'image_width' in self.args["generator_params"]:
            self.args["generator_params"]["image_width"] = self.args["generator_params"]["image_width"]
        if 'total_frames' in self.args["generator_params"]:
            self.args["generator_params"]["total_frames"] = self.args["generator_params"]["total_frames"]
        if 'raster_phase' in self.args["generator_params"]:
            self.args["generator_params"]["raster_phase"] = self.args["generator_params"]["raster_phase"]
        if 'fill_fraction' in self.args["generator_params"]:
            self.args["generator_params"]["fill_fraction"] = self.args["generator_params"]["fill_fraction"]
        if 'y_shifts' in self.args["generator_params"]:
            self.args["generator_params"]["y_shifts"] = self.args["generator_params"]["y_shifts"]
        if 'x_shifts' in self.args["generator_params"]:
            self.args["generator_params"]["x_shifts"] = self.args["generator_params"]["x_shifts"]
        # To be removed once fully transitioned to CLI
        self.args["generator_params"]["train_path"] = self.args["generator_params"][
            "data_path"
        ]
        self.args["test_generator_params"]["train_path"] = self.args[
            "test_generator_params"
        ]["data_path"]

        # Forward parameters to the training agent
        self.args["finetuning_params"]["steps_per_epoch"] = self.args[
            "finetuning_params"
        ]["steps_per_epoch"]

        self.args["finetuning_params"]["batch_size"] = self.args["generator_params"][
            "batch_size"
        ]

        # This is used to send to the legacy parameter tracking system
        # to specify each sub-object type.
        self.args["generator_params"]["type"] = "generator"
        self.args["test_generator_params"]["type"] = "generator"
        self.args["finetuning_params"]["type"] = "trainer"

        # save the json parameters to 2 different files
        finetuning_json_path = outdir / f"{uid}_finetuning.json"
        with open(finetuning_json_path, "w") as f:
            json.dump(self.args["finetuning_params"], f, indent=2)
        self.logger.info(f"wrote {finetuning_json_path}")

        generator_json_path = outdir / f"{uid}_generator.json"
        with open(generator_json_path, "w") as f:
            json.dump(self.args["generator_params"], f, indent=2)
        self.logger.info(f"wrote {generator_json_path}")

        test_generator_json_path = outdir / f"{uid}_test_generator.json"
        with open(test_generator_json_path, "w") as f:
            json.dump(self.args["test_generator_params"], f, indent=2)
        self.logger.info(f"wrote {test_generator_json_path}")

        generator_obj = ClassLoader(generator_json_path)
        data_generator = generator_obj.find_and_build()(generator_json_path)

        test_generator_obj = ClassLoader(test_generator_json_path)
        data_test_generator = test_generator_obj.find_and_build()(
            test_generator_json_path
        )

        finetuning_obj = ClassLoader(finetuning_json_path)

        training_class = finetuning_obj.find_and_build()(
            data_generator, data_test_generator, finetuning_json_path
        )

        self.logger.info("created objects for training")
        training_class.run()

        self.logger.info("fine tuning job finished - finalizing output model")
        training_class.finalize()


if __name__ == "__main__":  # pragma: nocover
    fine = FineTuning()
    fine.run()
