package co.immanuel.arkspeech.core.config

import co.immanuel.arkspeech.core.asr.RoutingTable
import co.immanuel.arkspeech.core.support.TestSupport
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class DefinitionRepositoryTest {
    private fun repo() = FileDefinitionRepository(Files.createTempDirectory("arkdefs").toFile(), TestSupport.clock)

    @Test fun `sample definitions validate`() {
        for (d in TestSupport.definitions()) assertEquals(emptyList(), DefinitionValidator.validate(d), d.id)
    }

    @Test fun `validator catches common mistakes`() {
        val d = TestSupport.definition("book_appointment")
        val bad = d.copy(
            id = "Bad Id",
            languages = listOf("xx"),
            triggers = d.triggers.copy(patterns = listOf("(unclosed")),
            fields = d.fields + d.fields.first().copy(),
            confidenceThreshold = 1.5f,
        )
        val errors = DefinitionValidator.validate(bad).map { it.path }
        assertTrue("id" in errors && "languages" in errors && "confidenceThreshold" in errors, "$errors")
        assertTrue(errors.any { it.contains("patterns") } && errors.any { it.contains("fields[5].name") }, "$errors")
    }

    @Test fun `saving creates versions and history`() {
        val r = repo()
        val d = TestSupport.definition("report_issue")
        val v1 = r.save(d, author = "alice", note = "first")
        assertEquals(1, v1.version)
        val v2 = r.save(v1.copy(confidenceThreshold = 0.6f), author = "bob", note = "raise threshold")
        assertEquals(2, v2.version)
        assertEquals(0.6f, r.get(d.id)!!.confidenceThreshold)
        assertEquals(0.5f, r.getVersion(d.id, 1)!!.confidenceThreshold)
        val h = r.history(d.id)
        assertEquals(listOf(2, 1), h.map { it.version })
        assertEquals("bob", h[0].updatedBy)
        assertEquals("raise threshold", h[0].changeNote)
    }

    @Test fun `disable and rollback are recorded as versions`() {
        val r = repo()
        val d = r.save(TestSupport.definition("stock_update"))
        val off = r.setEnabled(d.id, false, "ops")!!
        assertFalse(off.enabled)
        assertEquals(2, off.version)
        assertTrue(r.activeFor("en").none { it.id == d.id })
        val back = r.rollback(d.id, 1, "ops")!!
        assertEquals(3, back.version)
        assertTrue(back.enabled)
        assertTrue(back.changeNote.contains("Rolled back"))
    }

    @Test fun `invalid definitions are refused`() {
        val r = repo()
        assertFailsWith<InvalidDefinitionException> { r.save(TestSupport.definition("stock_update").copy(fields = emptyList(), triggers = IntentTriggers())) }
    }

    @Test fun `export and import round trip`() {
        val a = repo()
        TestSupport.definitions().forEach { a.save(it) }
        a.saveRouting(RoutingTable("omni", mapOf("hi" to "indic-hi")))
        val bundle = a.exportBundle("test")
        val json = ConfigJson.pretty.encodeToString(ConfigBundle.serializer(), bundle)
        val parsed = ConfigJson.pretty.decodeFromString(ConfigBundle.serializer(), json)
        val b = repo()
        val report = b.importBundle(parsed, author = "importer")
        assertTrue(report.ok, "${report.errors}")
        assertEquals(4, b.list().size)
        assertEquals("indic-hi", b.routing()!!.byLanguage["hi"])
        assertNotNull(b.get("book_appointment"))
        val badBundle = parsed.copy(definitions = parsed.definitions + parsed.definitions.first())
        assertFalse(b.importBundle(badBundle).ok)
    }

    @Test fun `active definitions filter by language and enabled`() {
        val r = repo()
        TestSupport.definitions().forEach { r.save(it) }
        val mr = r.activeFor("mr").map { it.id }
        assertTrue("record_payment" !in mr && "report_issue" in mr, "$mr")
    }
}
