import logging
from random import choices

from pytorch_lightning import seed_everything

from hakubooru.dataset import load_db, Post
from hakubooru.caption import KohakuCaptioner
from hakubooru.export import Exporter, FileSaver
from hakubooru.logging import logger
from hakubooru.source import TarSource


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.info("Loading danbooru2023.db")
    load_db("datasets/danbooru2023/metadata/danbooru2023.db")

    num_samples = 10


    logger.info("Querying posts")
    # Querying posts for:
    # All the post after 5_000_000
    # 1/2 of the post before 5_000_000, after 3_000_000
    # 1/3 of the post before 3_000_000
    # Use seed_everything(1) to make the result reproducible
    seed_everything(1)
    member_choosed_post = (
        list(Post.select().where(Post.id >= 5_000_000))
        + choices(
            Post.select().where(Post.id < 5_000_000, Post.id >= 3_000_000), k=1_000_000
        )
        + choices(Post.select().where(Post.id < 3_000_000), k=1_000_000)
    )
    nonmember_choosed_post = (
        [item for item in list(Post.select().where(Post.id < 5_000_000)) if item not in member_choosed_post]
    )

    logger.info(f"Build exporter for members")
    exporter = Exporter(
        source=TarSource("datasets/danbooru2023/data"),
        saver=FileSaver("datasets/danbooru2023/images_member"),
        captioner=KohakuCaptioner(),
        process_batch_size=100000,
        process_threads=2,
    )
    logger.info(f"Found {len(member_choosed_post)} posts")
    logger.info(f"Exporting images for members")
    exporter.export_posts(member_choosed_post)

    logger.info(f"Build exporter for non-members")
    exporter = Exporter(
        source=TarSource("datasets/danbooru2023/data"),
        saver=FileSaver("datasets/danbooru2023/images_nonmember"),
        captioner=KohakuCaptioner(),
        process_batch_size=100000,
        process_threads=2,
    )
    logger.info(f"Found {len(nonmember_choosed_post)} posts")
    logger.info(f"Exporting images for non-members")
    exporter.export_posts(nonmember_choosed_post)

    